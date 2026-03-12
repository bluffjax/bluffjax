import datetime
from typing import Any, Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax import make
from bluffjax.environments.leduc_holdem.leduc_holdem import LeducHoldemState
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils.leduc_exploitability import (
    _INFOSET_LIST,
    exploitability_jit,
    policy_array_from_network,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

_exploitability_jitted = jax.jit(exploitability_jit)


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    absorbing: BoolArray
    done: BoolArray
    value: FloatArray
    player_idx: IntArray
    info: dict[str, Any]


class RunnerState(NamedTuple):
    train_state: TrainState
    state: LeducHoldemState
    obs: FloatArray
    done: BoolArray
    update_step: IntArray
    rng: PRNGKeyArray


class UpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("leduc_holdem", **config["env_kwargs"])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    def linear_decay(count: int) -> float:
        frac = 1.0 - (count // config["num_gradient_steps"])
        return config["lr"] * frac

    def wandb_callback(exp_id: int, metrics: dict, info: dict) -> None:
        metrics.update(info)
        np_log_dict = {k: np.array(v) for k, v in metrics.items()}
        LOGGER.log(int(exp_id), np_log_dict)

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng: PRNGKeyArray,
        ) -> tuple[TrainState, ActorCriticDiscreteMLP, LeducHoldemState, FloatArray]:
            rng, rng_reset = jax.random.split(rng)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng, rng_network_init = jax.random.split(rng)
            network = ActorCriticDiscreteMLP(
                action_dim=env.num_actions, hidden_dim=config["fc_dim_size"]
            )
            network_params = network.init(rng_network_init, obs)

            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_decay, eps=1e-5),
            )
            train_state = TrainState.create(
                apply_fn=network.apply, params=network_params, tx=tx
            )
            return (train_state, network, state, obs)

        rng, rng_setup = jax.random.split(rng)
        (train_state, network, state, obs) = train_setup(rng_setup)

        def update_step(
            runner_state: RunnerState, unused: None
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state: RunnerState, unused: None
            ) -> tuple[RunnerState, Transition]:
                train_state = runner_state.train_state
                state = runner_state.state
                obs = runner_state.obs
                rng = runner_state.rng

                rng, rng_action, rng_step = jax.random.split(rng, 3)

                action_logits, value = network.apply(train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                logits_masked = jnp.where(action_mask, action_logits, -jnp.inf)
                pi = distrax.Categorical(logits=logits_masked)
                action = pi.sample(seed=rng_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                log_prob = pi.log_prob(action)
                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    absorbing=absorbing,
                    done=done,
                    value=value,
                    player_idx=state.current_player_idx,
                    info=info,
                )
                runner_state = RunnerState(
                    train_state=train_state,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step, runner_state, None, config["num_steps_per_env_per_update"]
            )

            train_state = runner_state.train_state
            last_obs = runner_state.obs
            last_state = runner_state.state
            rng, rng_update = jax.random.split(runner_state.rng)

            _, last_val = network.apply(train_state.params, last_obs)

            def calculate_gae(
                transitions: Transition,
                last_val: FloatArray,
                last_player_idx: IntArray,
            ) -> tuple[FloatArray, FloatArray]:
                num_envs = transitions.done.shape[1]
                num_agents = transitions.reward.shape[-1]

                next_value = jnp.zeros((num_envs, num_agents))
                next_value = next_value.at[jnp.arange(num_envs), last_player_idx].set(
                    last_val
                )
                pending_reward = transitions.reward[-1]
                gae_carry = jnp.zeros((num_envs, num_agents))

                def _gather_acting(arr: FloatArray, acting: IntArray) -> FloatArray:
                    return jnp.take_along_axis(
                        arr, acting[..., None].astype(jnp.int32), axis=-1
                    ).squeeze(-1)

                def get_advantages(carry, transition: Transition):
                    next_value, pending_reward, gae_carry = carry
                    acting = transition.player_idx
                    value = transition.value
                    reward = transition.reward
                    absorbing = transition.absorbing
                    done = transition.done

                    next_value = jnp.where(
                        done[:, None], jnp.zeros_like(next_value), next_value
                    )
                    gae_carry = jnp.where(
                        done[:, None], jnp.zeros_like(gae_carry), gae_carry
                    )

                    pending_reward_acting = _gather_acting(pending_reward, acting)
                    own_reward_acting = _gather_acting(reward, acting)
                    reward_acting = jnp.where(
                        done,
                        own_reward_acting,
                        pending_reward_acting + own_reward_acting,
                    )

                    next_val_acting = _gather_acting(next_value, acting)
                    absorbing_acting = _gather_acting(absorbing, acting)
                    delta = (
                        reward_acting
                        + config["gamma"] * next_val_acting * (1 - absorbing_acting)
                        - value
                    )

                    gae_prev_acting = _gather_acting(gae_carry, acting)
                    gae_new = (
                        delta
                        + config["gamma"]
                        * config["gae_lambda"]
                        * (1 - done)
                        * gae_prev_acting
                    )

                    next_value = next_value.at[jnp.arange(num_envs), acting].set(value)
                    gae_carry = gae_carry.at[jnp.arange(num_envs), acting].set(gae_new)

                    pending_reward = jnp.where(done[:, None], reward, pending_reward)
                    clear_mask = jnp.arange(num_agents) == acting[:, None]
                    pending_reward = jnp.where(
                        clear_mask, jnp.zeros_like(reward), pending_reward
                    )
                    return (next_value, pending_reward, gae_carry), gae_new

                init_carry = (next_value, pending_reward, gae_carry)
                _, advantages = jax.lax.scan(
                    get_advantages,
                    init_carry,
                    transitions,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + transitions.value

            last_player_idx = last_state.current_player_idx
            advantages, targets = calculate_gae(transitions, last_val, last_player_idx)

            def update_epoch(
                update_state: UpdateState, unused: None
            ) -> tuple[UpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(update_state.rng)
                batch = (
                    update_state.transitions,
                    update_state.advantages,
                    update_state.targets,
                )

                def _reshape_batch(x):
                    if x.ndim == 2:
                        return x.reshape(-1)
                    elif x.ndim == 3:
                        return x.reshape(-1, *x.shape[2:])
                    else:
                        return x.reshape(-1, *x.shape[3:])

                batch_reshaped = jax.tree_util.tree_map(_reshape_batch, batch)

                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch_reshaped
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape(config["num_minibatches"], -1, *x.shape[1:]),
                    batch_shuffled,
                )

                def update_minibatch(
                    train_state: TrainState,
                    minibatch: tuple[Transition, FloatArray, FloatArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    transitions_mb, advantages_mb, targets_mb = minibatch

                    def loss(
                        params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                    ) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, value = network.apply(params, transitions_mb.obs)
                        logits_masked = jnp.where(
                            transitions_mb.action_mask, logits, -jnp.inf
                        )
                        pi = distrax.Categorical(logits=logits_masked)
                        log_prob = pi.log_prob(transitions_mb.action)

                        logratio = log_prob - transitions_mb.log_prob
                        ratio = jnp.exp(logratio)
                        gae_normalized = (advantages_mb - advantages_mb.mean()) / (
                            advantages_mb.std() + 1e-8
                        )
                        loss_actor = ratio * gae_normalized
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        loss_actor = -jnp.minimum(loss_actor, loss_actor_clipped).mean()
                        entropy = pi.entropy().mean()

                        value_clipped = transitions_mb.value + jnp.clip(
                            value - transitions_mb.value,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - targets_mb),
                                jnp.square(value_clipped - targets_mb),
                            ).mean()
                        )

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        kl_backward = ((ratio - 1) - logratio).mean()
                        kl_forward = (ratio * logratio - (ratio - 1)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["clip_eps"])

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": ratio.mean(),
                            "ratio_min": ratio.min(),
                            "ratio_max": ratio.max(),
                            "gae_mean": advantages_mb.mean(),
                            "gae_std": advantages_mb.std(),
                            "mean_target": targets_mb.mean(),
                            "value_pred_mean": value.mean(),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                        }

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (_, aux), grads = grad_fn(
                        train_state.params,
                        transitions_mb,
                        advantages_mb,
                        targets_mb,
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)

                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch, update_state.train_state, minibatches
                )

                update_state = UpdateState(
                    train_state=final_train_state,
                    transitions=update_state.transitions,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )

                return update_state, batch_stats

            update_state = UpdateState(
                train_state=train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
                rng=rng_update,
            )

            final_update_state, loss_info = lax.scan(
                update_epoch, update_state, None, config["num_epochs"]
            )
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)

            returns_f32 = transitions.info["returns"].astype(jnp.float32)
            masked_returns = jnp.where(
                transitions.done[..., None], returns_f32, jnp.float32(0.0)
            )
            done_count = jnp.maximum(transitions.done.sum(), 1)
            returns_avg = masked_returns.sum() / done_count
            returns_avg_agent_one = masked_returns[..., 0].sum() / done_count
            returns_avg_agent_two = masked_returns[..., 1].sum() / done_count

            masked_timesteps = jnp.where(
                transitions.done,
                transitions.info["timestep"].astype(jnp.float32),
                jnp.float32(0.0),
            )
            ep_length_avg = masked_timesteps.sum() / done_count

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
                "update_step": runner_state.update_step,
            }
            metric.update(loss_info)

            def logging_callback(seed_val, metric_dict, info, params):
                policy_arr = policy_array_from_network(
                    network.apply, params, _INFOSET_LIST
                )
                expl = float(_exploitability_jitted(policy_arr))
                metric_dict = dict(metric_dict)
                metric_dict["exploitability"] = expl
                wandb_callback(seed_val, metric_dict, info)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                metric,
                transitions.info,
                final_update_state.train_state.params,
            )

            runner_state = RunnerState(
                train_state=final_update_state.train_state,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            train_state=train_state,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step, initial_runner_state, None, config["num_update_steps"]
        )

        return final_runner_state

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ppo")
def main(config: dict) -> None:
    try:
        config = OmegaConf.to_container(config)
        config["num_update_steps"] = (
            config["num_timesteps"]
            // config["num_envs"]
            // config["num_steps_per_env_per_update"]
        )
        config["num_gradient_steps"] = (
            config["num_update_steps"]
            * config["num_epochs"]
            * config["num_minibatches"]
        )

        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_vjit = jax.block_until_ready(jax.jit(jax.vmap(make_train(config))))
        print("Compile finished...")

        job_type = f"{config['job_type']}_{config['env_name']}"
        group = f"{config['env_name']}" + datetime.datetime.now().strftime(
            "_%Y-%m-%d_%H-%M-%S"
        )
        global LOGGER
        LOGGER = WandbMultiLogger(
            project=config["project"],
            group=group,
            job_type=job_type,
            config=config,
            mode=(lambda: "online" if config["wandb"] else "disabled")(),
            seed=config["seed"],
            num_seeds=config["num_seeds"],
        )

        print("Running...")
        _ = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
