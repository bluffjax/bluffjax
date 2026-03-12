import datetime
from typing import Any, Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax import make
from bluffjax.environments.leduc_holdem.leduc_holdem import LeducHoldemState
from bluffjax.networks.mlp import QNetworkDiscreteMLP
from bluffjax.utils.game_utils.leduc_exploitability import (
    _INFOSET_LIST,
    exploitability_jit,
    policy_array_from_qnetwork,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

_exploitability_jitted = jax.jit(exploitability_jit)


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    reward: FloatArray
    absorbing: BoolArray
    done: BoolArray
    q_val: FloatArray
    player_idx: IntArray
    next_obs: FloatArray
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

    def epsilon_schedule(update_step: FloatArray) -> FloatArray:
        decay_steps = config["exploration_fraction"] * config["num_update_steps"]
        frac = jnp.minimum(1.0, update_step.astype(jnp.float32) / decay_steps)
        return config["start_e"] + frac * (config["end_e"] - config["start_e"])

    def wandb_callback(exp_id: int, metrics: dict, info: dict) -> None:
        metrics.update(info)
        np_log_dict = {k: np.array(v) for k, v in metrics.items()}
        LOGGER.log(int(exp_id), np_log_dict)

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng: PRNGKeyArray,
        ) -> tuple[TrainState, QNetworkDiscreteMLP, LeducHoldemState, FloatArray]:
            rng, rng_reset = jax.random.split(rng)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng, rng_network_init = jax.random.split(rng)
            network = QNetworkDiscreteMLP(
                action_dim=env.num_actions, hidden_dim=config["fc_dim_size"]
            )
            network_params = network.init(rng_network_init, obs)

            tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.radam(learning_rate=linear_decay),
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

                q_vals = network.apply(train_state.params, obs.astype(jnp.float32))
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)

                eps = epsilon_schedule(
                    jnp.array(runner_state.update_step, dtype=jnp.float32)
                )
                rng_uniform, rng_random = jax.random.split(rng_action, 2)
                explore = jax.random.uniform(rng_uniform, (config["num_envs"],)) < eps
                random_probs = action_mask.astype(jnp.float32) / (
                    action_mask.sum(axis=-1, keepdims=True) + 1e-8
                )
                rng_randoms = jax.random.split(rng_random, config["num_envs"])
                random_action = jax.vmap(
                    lambda rng_val, p: distrax.Categorical(probs=p).sample(seed=rng_val)
                )(rng_randoms, random_probs)
                action = jnp.where(explore, random_action, greedy_action)

                chosen_q = jnp.take_along_axis(
                    q_vals, action[..., None].astype(jnp.int32), axis=-1
                ).squeeze(-1)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                reward_acting = jnp.take_along_axis(
                    reward,
                    state.current_player_idx[..., None].astype(jnp.int32),
                    axis=-1,
                ).squeeze(-1)

                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    reward=reward_acting,
                    absorbing=absorbing,
                    done=done,
                    q_val=chosen_q,
                    player_idx=state.current_player_idx,
                    next_obs=next_obs,
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
            rng, rng_update = jax.random.split(runner_state.rng)

            last_q_raw = jnp.max(
                network.apply(train_state.params, last_obs.astype(jnp.float32)), axis=-1
            )
            last_q = -last_q_raw

            def compute_q_lambda_targets(
                transitions: Transition, last_q: FloatArray
            ) -> FloatArray:
                reward = transitions.reward
                next_obs = transitions.next_obs.astype(jnp.float32)
                done = transitions.done
                q_vals_next = network.apply(train_state.params, next_obs)

                done_f32 = done.astype(jnp.float32)
                lambda_returns_last = (
                    reward[-1] + config["gamma"] * (1.0 - done_f32[-1]) * last_q
                )
                idx = jnp.maximum(0, q_vals_next.shape[0] - 2)
                last_max_q = -jnp.max(q_vals_next[idx], axis=-1)

                def _get_target(carry, rew_q_done):
                    reward_t, q_next_t, done_t = rew_q_done
                    lambda_returns, next_q = carry
                    target_bootstrap = (
                        reward_t
                        + config["gamma"] * (1.0 - done_t.astype(jnp.float32)) * next_q
                    )
                    delta = -lambda_returns - next_q
                    lambda_returns_new = (
                        target_bootstrap + config["gamma"] * config["q_lambda"] * delta
                    )
                    lambda_returns_new = (
                        1.0 - done_t.astype(jnp.float32)
                    ) * lambda_returns_new + done_t.astype(jnp.float32) * reward_t
                    next_q_new = -jnp.max(q_next_t, axis=-1)
                    return (lambda_returns_new, next_q_new), lambda_returns_new

                init_carry = (lambda_returns_last, last_max_q)
                reward_prev = reward[:-1]
                q_next_prev = q_vals_next[:-1]
                done_prev = done[:-1]
                (_, _), targets_prev = lax.scan(
                    _get_target,
                    init_carry,
                    (reward_prev, q_next_prev, done_prev),
                    reverse=True,
                )
                targets = jnp.concatenate(
                    [targets_prev, lambda_returns_last[None, ...]], axis=0
                )
                return targets

            targets = compute_q_lambda_targets(transitions, last_q)

            def update_epoch(
                update_state: UpdateState, unused: None
            ) -> tuple[UpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(update_state.rng)
                batch = (update_state.transitions, update_state.targets)

                def _reshape_batch(x):
                    if x.ndim == 2:
                        return x.reshape(-1)
                    elif x.ndim == 3:
                        return x.reshape(-1, *x.shape[2:])
                    else:
                        return x.reshape(-1, *x.shape[3:])

                def _reshape_transition(trans: Transition) -> Transition:
                    info_reshaped = {
                        k: _reshape_batch(v) for k, v in trans.info.items()
                    }
                    return Transition(
                        obs=_reshape_batch(trans.obs),
                        action_mask=_reshape_batch(trans.action_mask),
                        action=_reshape_batch(trans.action),
                        reward=_reshape_batch(trans.reward),
                        absorbing=_reshape_batch(trans.absorbing),
                        done=_reshape_batch(trans.done),
                        q_val=_reshape_batch(trans.q_val),
                        player_idx=_reshape_batch(trans.player_idx),
                        next_obs=_reshape_batch(trans.next_obs),
                        info=info_reshaped,
                    )

                batch_reshaped = (
                    _reshape_transition(update_state.transitions),
                    _reshape_batch(update_state.targets),
                )

                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                trans_reshaped, targets_reshaped = batch_reshaped
                info_shuffled = {
                    k: jnp.take(v, permutation, axis=0)
                    for k, v in trans_reshaped.info.items()
                }
                trans_shuffled = Transition(
                    obs=jnp.take(trans_reshaped.obs, permutation, axis=0),
                    action_mask=jnp.take(
                        trans_reshaped.action_mask, permutation, axis=0
                    ),
                    action=jnp.take(trans_reshaped.action, permutation, axis=0),
                    reward=jnp.take(trans_reshaped.reward, permutation, axis=0),
                    absorbing=jnp.take(trans_reshaped.absorbing, permutation, axis=0),
                    done=jnp.take(trans_reshaped.done, permutation, axis=0),
                    q_val=jnp.take(trans_reshaped.q_val, permutation, axis=0),
                    player_idx=jnp.take(trans_reshaped.player_idx, permutation, axis=0),
                    next_obs=jnp.take(trans_reshaped.next_obs, permutation, axis=0),
                    info=info_shuffled,
                )
                targets_shuffled = jnp.take(targets_reshaped, permutation, axis=0)

                batch_size = config["batch_shuffle_dim"]
                minibatch_size = batch_size // config["num_minibatches"]
                info_minibatches = {
                    k: (
                        v.reshape(
                            config["num_minibatches"], minibatch_size, *v.shape[1:]
                        )
                        if v.ndim > 1
                        else v.reshape(config["num_minibatches"], minibatch_size)
                    )
                    for k, v in trans_shuffled.info.items()
                }

                minibatches_trans = Transition(
                    obs=trans_shuffled.obs.reshape(
                        config["num_minibatches"], minibatch_size, -1
                    ),
                    action_mask=trans_shuffled.action_mask.reshape(
                        config["num_minibatches"], minibatch_size, -1
                    ),
                    action=trans_shuffled.action.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
                    reward=trans_shuffled.reward.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
                    absorbing=trans_shuffled.absorbing.reshape(
                        config["num_minibatches"], minibatch_size, -1
                    ),
                    done=trans_shuffled.done.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
                    q_val=trans_shuffled.q_val.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
                    player_idx=trans_shuffled.player_idx.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
                    next_obs=trans_shuffled.next_obs.reshape(
                        config["num_minibatches"], minibatch_size, -1
                    ),
                    info=info_minibatches,
                )
                minibatches_targets = targets_shuffled.reshape(
                    config["num_minibatches"], minibatch_size
                )

                def update_minibatch(
                    train_state: TrainState,
                    minibatch: tuple[Transition, FloatArray],
                ) -> tuple[TrainState, dict[str, FloatArray]]:
                    trans, targets_mb = minibatch

                    def loss_fn(params, trans, targets_mb):
                        q_vals = network.apply(params, trans.obs.astype(jnp.float32))
                        chosen_q = jnp.take_along_axis(
                            q_vals,
                            trans.action[..., None].astype(jnp.int32),
                            axis=-1,
                        ).squeeze(-1)
                        td_loss = 0.5 * jnp.square(chosen_q - targets_mb).mean()
                        return td_loss, {
                            "td_loss": td_loss,
                            "q_values": chosen_q.mean(),
                        }

                    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        train_state.params, trans, targets_mb
                    )
                    aux["grad_norm"] = pytree_norm(grads)
                    updated_train_state = train_state.apply_gradients(grads=grads)
                    return updated_train_state, aux

                minibatches = (minibatches_trans, minibatches_targets)
                final_train_state, batch_stats = lax.scan(
                    update_minibatch,
                    update_state.train_state,
                    minibatches,
                )

                update_state = UpdateState(
                    train_state=final_train_state,
                    transitions=update_state.transitions,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, batch_stats

            update_state = UpdateState(
                train_state=train_state,
                transitions=transitions,
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

            td_loss = jnp.mean(jnp.square(transitions.q_val - targets))
            q_values_mean = jnp.mean(transitions.q_val)
            eps = epsilon_schedule(
                jnp.array(runner_state.update_step, dtype=jnp.float32)
            )

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
                "td_loss": td_loss,
                "q_values_rollout": q_values_mean,
                "epsilon": eps,
                "update_step": runner_state.update_step,
            }
            metric.update(loss_info)

            def logging_callback(seed_val, metric_dict, info, params):
                policy_arr = policy_array_from_qnetwork(
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


@hydra.main(version_base=None, config_path="./", config_name="config_pqn")
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
