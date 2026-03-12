import jax

import datetime
from typing import Any, Callable, NamedTuple
import distrax
from flax.training.train_state import TrainState
import hydra
from jax import lax, tree_util
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
from bluffjax import make
from bluffjax.environments.kuhn_poker.kuhn_poker import KuhnState
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.jax_utils import jprint, pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger
from bluffjax.utils.typing import (
    Any,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)
from bluffjax.utils.game_utils.kuhn_exploitability import (
    exploitability,
    policy_from_bluffjax_keys,
    policy_from_network,
)


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    log_prob: FloatArray
    reward: FloatArray
    absorbing: BoolArray
    done: bool
    value: FloatArray
    player_idx: IntArray
    info: dict[str, Any]


class RunnerState(NamedTuple):
    train_state: TrainState
    state: KuhnState
    obs: FloatArray
    done: bool
    update_step: int
    rng: PRNGKeyArray


class Updatestate(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("kuhn_poker", **config["env_kwargs"])

    # config, network & optimizer (one transition per step per env in AEC)
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    def linear_decay(count: int) -> float:
        # decays from 1.0 to 0 over num_gradient_steps
        frac = 1.0 - (count // config["num_gradient_steps"])
        return config["lr"] * frac

    def wandb_callback(exp_id: int, metrics: dict, info: dict) -> None:
        metrics.update(info)
        np_log_dict = {k: np.array(v) for k, v in metrics.items()}
        LOGGER.log(int(exp_id), np_log_dict)

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng: PRNGKeyArray,
        ) -> tuple[TrainState, ActorCriticDiscreteMLP, KuhnState, FloatArray]:
            # env reset
            rng, rng_reset = jax.random.split(rng)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            # network
            rng, rng_network_init = jax.random.split(rng)
            network = ActorCriticDiscreteMLP(
                action_dim=2, hidden_dim=config["fc_dim_size"]
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

        # update loop
        def update_step(
            runner_state: RunnerState, unused: None
        ) -> tuple[RunnerState, None]:

            # collect samples
            def step(
                runner_state: RunnerState, unused: None
            ) -> tuple[RunnerState, Transition]:
                train_state = runner_state.train_state
                state = runner_state.state
                obs = runner_state.obs
                rng = runner_state.rng

                (
                    rng,
                    rng_action,
                    rng_step,
                    rng_reset,
                ) = jax.random.split(rng, 4)

                # select actions
                action_logits, value = network.apply(train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                logits_mask = jnp.where(action_mask, action_logits, -jnp.inf)
                pi = distrax.Categorical(logits=logits_mask)
                action = pi.sample(seed=rng_action)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                # store transition
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
                """Compute GAE separately for each agent. Terminal rewards are
                propagated to all agents regardless of when they last acted."""
                num_envs = transitions.done.shape[1]
                num_agents = transitions.reward.shape[-1]

                # Bootstrap: next_value for the agent who would act after last step
                next_value = jnp.zeros((num_envs, num_agents))
                next_value = next_value.at[jnp.arange(num_envs), last_player_idx].set(
                    last_val
                )

                pending_reward = transitions.reward[-1]  # seed with final rewards
                gae_carry = jnp.zeros((num_envs, num_agents))

                def _gather_acting(arr: FloatArray, acting: IntArray) -> FloatArray:
                    """Index (num_envs, num_agents) array by acting agent per env."""
                    return jnp.take_along_axis(
                        arr, acting[..., None].astype(jnp.int32), axis=-1
                    ).squeeze(-1)

                def get_advantages(carry, transition: Transition):
                    next_value, pending_reward, gae_carry = carry
                    acting = transition.player_idx  # (num_envs,)
                    value = transition.value  # (num_envs,)
                    reward = transition.reward  # (num_envs, num_agents)
                    absorbing = transition.absorbing  # (num_envs, num_agents)
                    done = transition.done  # (num_envs,)

                    # Reset next_value and gae_carry at episode boundaries
                    next_value = jnp.where(
                        done[:, None], jnp.zeros_like(next_value), next_value
                    )
                    gae_carry = jnp.where(
                        done[:, None], jnp.zeros_like(gae_carry), gae_carry
                    )

                    # Reward for delta: pending (from future terminal) + own reward at this step.
                    # When done: acting agent's action was terminal, so pending == own (avoid double-count).
                    pending_reward_acting = _gather_acting(pending_reward, acting)
                    own_reward_acting = _gather_acting(reward, acting)
                    reward_acting = jnp.where(
                        done,
                        own_reward_acting,
                        pending_reward_acting + own_reward_acting,
                    )

                    # Next value for acting agent (bootstrap for non-terminal)
                    next_val_acting = _gather_acting(next_value, acting)
                    absorbing_acting = _gather_acting(absorbing, acting)
                    delta = (
                        reward_acting
                        + config["gamma"] * next_val_acting * (1 - absorbing_acting)
                        - value
                    )

                    # GAE recursion: gae_t = delta_t + (gamma * gae_lambda) * gae_{t+1}
                    gae_prev_acting = _gather_acting(gae_carry, acting)
                    gae_new = (
                        delta
                        + config["gamma"]
                        * config["gae_lambda"]
                        * (1 - done)
                        * gae_prev_acting
                    )

                    # Update next_value and gae_carry for bootstrapping
                    next_value = next_value.at[jnp.arange(num_envs), acting].set(value)
                    gae_carry = gae_carry.at[jnp.arange(num_envs), acting].set(gae_new)

                    # At terminal: store reward for non-acting agent to propagate
                    # When not terminal and acting acts: clear pending for acting (we used it)
                    pending_reward = jnp.where(
                        done[:, None],
                        reward,  # Store all; non-acting agent's reward will be used when we see them
                        pending_reward,
                    )
                    # Clear pending for acting agent (we consumed it in delta)
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

            # update_network
            def update_epoch(update_state: Updatestate, unused: None) -> Updatestate:
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
                    transitions, advantages, targets = minibatch

                    def loss(
                        params, transitions, advantages, targets
                    ) -> tuple[FloatArray, dict[str, FloatArray]]:
                        # rerun network
                        logits, value = network.apply(params, transitions.obs)
                        logits_mask = jnp.where(
                            transitions.action_mask, logits, -jnp.inf
                        )
                        pi = distrax.Categorical(logits=logits_mask)
                        log_prob = pi.log_prob(transitions.action)

                        # actor loss
                        logratio = log_prob - transitions.log_prob
                        ratio = jnp.exp(logratio)
                        gae_normalized = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
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

                        # critic loss (clipped to prevent value overshooting)
                        value_clipped = transitions.value + jnp.clip(
                            value - transitions.value,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - targets),
                                jnp.square(value_clipped - targets),
                            ).mean()
                        )

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        # stats
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
                            "gae_mean": advantages.mean(),
                            "gae_std": advantages.std(),
                            "mean_target": targets.mean(),
                            "value_pred_mean": value.mean(),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                        }

                    grad_fn = jax.value_and_grad(loss, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, transitions, advantages, targets
                    )
                    aux["grad_norm"] = pytree_norm(grads)

                    updated_train_state = train_state.apply_gradients(grads=grads)

                    return updated_train_state, aux

                final_train_state, batch_stats = lax.scan(
                    update_minibatch, update_state.train_state, minibatches
                )

                update_state = Updatestate(
                    train_state=final_train_state,
                    transitions=transitions,
                    advantages=advantages,
                    targets=targets,
                    rng=rng,
                )

                return update_state, batch_stats

            update_state = Updatestate(
                train_state=train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
                rng=rng_update,
            )

            final_update_state, loss_info = lax.scan(
                update_epoch, update_state, None, config["num_epochs"]
            )

            # logging
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)

            masked_returns = jnp.where(
                transitions.done[..., None], transitions.info["returns"], 0.0
            )
            done_count = jnp.maximum(transitions.done.sum(), 1)
            returns_avg = masked_returns.sum() / done_count
            returns_avg_agent_one = masked_returns[..., 0].sum() / done_count
            returns_avg_agent_two = masked_returns[..., 1].sum() / done_count

            masked_timesteps = jnp.where(
                transitions.done, transitions.info["timestep"], 0.0
            )
            ep_length_avg = masked_timesteps.sum() / done_count

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
            }
            metric["update_step"] = runner_state.update_step
            metric.update(loss_info)

            def logging_callback(seed_val, metric_dict, probs, info):
                policy_net = policy_from_network(
                    network.apply, final_update_state.train_state.params
                )
                expl = exploitability(policy_net)
                metric_dict = dict(metric_dict)
                metric_dict["exploitability"] = float(expl)
                wandb_callback(seed_val, metric_dict, info)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                metric,
                transitions.info,
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
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool),
            update_step=0,
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
        # config
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

        # rng
        rng = jax.random.PRNGKey(config["seed"])
        rng_seeds = jax.random.split(rng, config["num_seeds"])
        exp_ids = jnp.arange(config["num_seeds"])

        print("Starting compile...")
        train_fn = make_train(config)
        train_vjit = jax.block_until_ready(jax.jit(jax.vmap(make_train(config))))
        print("Compile finished...")

        # wandb (sort by job_type then group)
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

        # run
        print("Running...")
        # out = jax.block_until_ready(train_fn(rng, 0))
        out_jit = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
