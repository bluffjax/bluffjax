import datetime
from typing import Callable, NamedTuple

import distrax
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import (
    Any,
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.kuhn_poker.kuhn_poker import KuhnState
from bluffjax.networks.mlp import ActorCriticDiscreteMLP
from bluffjax.utils.game_utils.kuhn_exploitability import (
    exploitability,
    policy_from_network,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    br_log_prob: FloatArray
    reward: FloatArray
    absorbing: BoolArray
    done: BoolArray
    value: FloatArray
    player_idx: IntArray
    is_br: BoolArray
    info: dict[str, Any]


class SLBufferState(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    seen: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    br_train_state: TrainState
    avg_train_state: TrainState
    sl_buffer: SLBufferState
    state: KuhnState
    obs: FloatArray
    done: BoolArray
    update_step: IntArray
    rng: PRNGKeyArray


class PPOUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
    advantages: FloatArray
    targets: FloatArray
    rng: PRNGKeyArray


class SLBatch(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray


class SLUpdateState(NamedTuple):
    train_state: TrainState
    sl_buffer: SLBufferState
    rng: PRNGKeyArray


def make_train(config: dict) -> Callable[[PRNGKeyArray, int], RunnerState]:
    env = make("kuhn_poker", **config["env_kwargs"])
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

    def init_sl_buffer(obs_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action_mask=jnp.zeros((capacity, 2), dtype=jnp.bool_),
            action=jnp.zeros((capacity,), dtype=jnp.int32),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def append_sl_samples(
        buffer: SLBufferState,
        obs_batch: FloatArray,
        action_mask_batch: BoolArray,
        action_batch: IntArray,
        valid_batch: BoolArray,
        rng: PRNGKeyArray,
    ) -> tuple[SLBufferState, PRNGKeyArray]:
        capacity = config["sl_reservoir_capacity"]

        def add_one(carry, sample):
            buffer, rng = carry
            obs, action_mask, action, valid = sample
            rng, rng_i = jax.random.split(rng)

            def _add(buf: SLBufferState) -> SLBufferState:
                k = buf.seen
                j = jax.random.randint(rng_i, (), 0, k + 1, dtype=jnp.int32)
                not_full = k < capacity
                write_idx = jnp.where(not_full, k, j)
                should_write = not_full | (j < capacity)

                def _write(b: SLBufferState) -> SLBufferState:
                    return SLBufferState(
                        obs=b.obs.at[write_idx].set(obs),
                        action_mask=b.action_mask.at[write_idx].set(action_mask),
                        action=b.action.at[write_idx].set(action),
                        seen=b.seen,
                        size=b.size,
                    )

                buf = lax.cond(should_write, _write, lambda b: b, buf)
                return SLBufferState(
                    obs=buf.obs,
                    action_mask=buf.action_mask,
                    action=buf.action,
                    seen=buf.seen + 1,
                    size=jnp.minimum(capacity, buf.size + 1),
                )

            buffer = lax.cond(valid, _add, lambda b: b, buffer)
            return (buffer, rng), None

        samples = (obs_batch, action_mask_batch, action_batch, valid_batch)
        (buffer, rng), _ = lax.scan(add_one, (buffer, rng), samples)
        return buffer, rng

    def sample_sl_batch(
        rng: PRNGKeyArray,
        buffer: SLBufferState,
        batch_size: int,
    ) -> SLBatch:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(rng, (batch_size,), 0, max_size, dtype=jnp.int32)
        return SLBatch(
            obs=buffer.obs[indices],
            action_mask=buffer.action_mask[indices],
            action=buffer.action[indices],
        )

    def masked_mean(x: FloatArray, mask: FloatArray) -> FloatArray:
        denom = jnp.maximum(mask.sum(), 1.0)
        return (x * mask).sum() / denom

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            ActorCriticDiscreteMLP,
            KuhnState,
            FloatArray,
            SLBufferState,
        ]:
            rng, rng_reset = jax.random.split(rng)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng, rng_network_init = jax.random.split(rng)
            network = ActorCriticDiscreteMLP(
                action_dim=2, hidden_dim=config["fc_dim_size"]
            )
            network_params = network.init(rng_network_init, obs)

            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=linear_decay, eps=1e-5),
            )
            avg_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
            )

            br_train_state = TrainState.create(
                apply_fn=network.apply, params=network_params, tx=br_tx
            )
            avg_train_state = TrainState.create(
                apply_fn=network.apply, params=network_params, tx=avg_tx
            )
            sl_buffer = init_sl_buffer(obs.shape[-1])
            return (br_train_state, avg_train_state, network, state, obs, sl_buffer)

        rng, rng_setup = jax.random.split(rng)
        (
            br_train_state,
            avg_train_state,
            network,
            state,
            obs,
            sl_buffer,
        ) = train_setup(rng_setup)

        def update_step(
            runner_state: RunnerState,
            unused: None,
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state: RunnerState,
                unused: None,
            ) -> tuple[RunnerState, Transition]:
                br_train_state = runner_state.br_train_state
                avg_train_state = runner_state.avg_train_state
                state = runner_state.state
                obs = runner_state.obs
                rng = runner_state.rng

                rng, rng_mix, rng_action, rng_step = jax.random.split(rng, 4)

                br_logits, value = network.apply(br_train_state.params, obs)
                avg_logits, _ = network.apply(avg_train_state.params, obs)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state)

                br_logits_masked = jnp.where(action_mask, br_logits, -jnp.inf)
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)

                is_br = jax.random.bernoulli(
                    rng_mix,
                    p=config["anticipatory_eta"],
                    shape=(config["num_envs"],),
                )
                acting_logits = jnp.where(
                    is_br[:, None], br_logits_masked, avg_logits_masked
                )

                acting_pi = distrax.Categorical(logits=acting_logits)
                br_pi = distrax.Categorical(logits=br_logits_masked)
                action = acting_pi.sample(seed=rng_action)
                br_log_prob = jnp.where(is_br, br_pi.log_prob(action), 0.0)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state, action)

                transition = Transition(
                    obs=obs,
                    action_mask=action_mask,
                    action=action,
                    br_log_prob=br_log_prob,
                    reward=reward,
                    absorbing=absorbing,
                    done=done,
                    value=value,
                    player_idx=state.current_player_idx,
                    is_br=is_br,
                    info=info,
                )
                runner_state = RunnerState(
                    br_train_state=br_train_state,
                    avg_train_state=avg_train_state,
                    sl_buffer=runner_state.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            runner_state, transitions = lax.scan(
                step,
                runner_state,
                None,
                config["num_steps_per_env_per_update"],
            )

            br_train_state = runner_state.br_train_state
            avg_train_state = runner_state.avg_train_state
            sl_buffer = runner_state.sl_buffer
            last_obs = runner_state.obs
            last_state = runner_state.state
            rng, rng_sl_append, rng_ppo_update, rng_sl_update = jax.random.split(
                runner_state.rng, 4
            )

            _, last_val = network.apply(br_train_state.params, last_obs)

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

            obs_flat = transitions.obs.reshape(-1, transitions.obs.shape[-1])
            action_mask_flat = transitions.action_mask.reshape(-1, 2)
            action_flat = transitions.action.reshape(-1)
            is_br_flat = transitions.is_br.reshape(-1)
            sl_buffer, rng_sl_append = append_sl_samples(
                sl_buffer,
                obs_flat,
                action_mask_flat,
                action_flat,
                is_br_flat,
                rng_sl_append,
            )

            def update_ppo_epoch(
                ppo_state: PPOUpdateState,
                unused: None,
            ) -> tuple[PPOUpdateState, dict[str, FloatArray]]:
                rng, rng_permute = jax.random.split(ppo_state.rng)
                batch = (
                    ppo_state.transitions,
                    ppo_state.advantages,
                    ppo_state.targets,
                )

                def _reshape_batch(x):
                    if x.ndim == 2:
                        return x.reshape(-1)
                    if x.ndim == 3:
                        return x.reshape(-1, *x.shape[2:])
                    return x.reshape(-1, *x.shape[3:])

                batch_reshaped = jax.tree_util.tree_map(_reshape_batch, batch)
                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
                batch_shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch_reshaped,
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

                        mask = transitions_mb.is_br.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)

                        mean_adv = masked_mean(advantages_mb, mask)
                        var_adv = masked_mean(
                            jnp.square(advantages_mb - mean_adv), mask
                        )
                        gae_normalized = (advantages_mb - mean_adv) / jnp.sqrt(
                            var_adv + 1e-8
                        )

                        logratio = log_prob - transitions_mb.br_log_prob
                        ratio = jnp.exp(logratio)
                        loss_actor_raw = ratio * gae_normalized
                        loss_actor_clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config["clip_eps"],
                                1.0 + config["clip_eps"],
                            )
                            * gae_normalized
                        )
                        actor_per_item = -jnp.minimum(
                            loss_actor_raw, loss_actor_clipped
                        )
                        loss_actor = (actor_per_item * mask).sum() / mask_denom

                        entropy = (pi.entropy() * mask).sum() / mask_denom

                        value_clipped = transitions_mb.value + jnp.clip(
                            value - transitions_mb.value,
                            -config["vf_clip"],
                            config["vf_clip"],
                        )
                        value_err = jnp.maximum(
                            jnp.square(value - targets_mb),
                            jnp.square(value_clipped - targets_mb),
                        )
                        value_loss = 0.5 * (value_err * mask).sum() / mask_denom

                        total_loss = (
                            loss_actor
                            + config["vf_coef"] * value_loss
                            - config["ent_coef"] * entropy
                        )

                        kl_backward = (
                            ((ratio - 1) - logratio) * mask
                        ).sum() / mask_denom
                        kl_forward = (
                            (ratio * logratio - (ratio - 1)) * mask
                        ).sum() / mask_denom
                        clip_frac = (
                            (jnp.abs(ratio - 1) > config["clip_eps"]).astype(
                                jnp.float32
                            )
                            * mask
                        ).sum() / mask_denom
                        br_sample_frac = mask.mean()

                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": loss_actor,
                            "entropy": entropy,
                            "ratio_mean": ratio.mean(),
                            "ratio_min": ratio.min(),
                            "ratio_max": ratio.max(),
                            "gae_mean": mean_adv,
                            "gae_std": jnp.sqrt(var_adv + 1e-8),
                            "mean_target": targets_mb.mean(),
                            "value_pred_mean": value.mean(),
                            "kl_backward": kl_backward,
                            "kl_forward": kl_forward,
                            "clip_frac": clip_frac,
                            "br_sample_frac_minibatch": br_sample_frac,
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
                    update_minibatch,
                    ppo_state.train_state,
                    minibatches,
                )
                ppo_state = PPOUpdateState(
                    train_state=final_train_state,
                    transitions=ppo_state.transitions,
                    advantages=ppo_state.advantages,
                    targets=ppo_state.targets,
                    rng=rng,
                )
                return ppo_state, batch_stats

            ppo_state = PPOUpdateState(
                train_state=br_train_state,
                transitions=transitions,
                advantages=advantages,
                targets=targets,
                rng=rng_ppo_update,
            )
            final_ppo_state, ppo_loss_info = lax.scan(
                update_ppo_epoch,
                ppo_state,
                None,
                config["num_epochs"],
            )
            br_train_state = final_ppo_state.train_state
            ppo_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), ppo_loss_info)

            def update_sl_step(
                sl_state: SLUpdateState,
                unused: None,
            ) -> tuple[SLUpdateState, dict[str, FloatArray]]:
                rng, rng_batch = jax.random.split(sl_state.rng)
                has_data = sl_state.sl_buffer.size > 0

                def _train_step(train_state: TrainState):
                    batch = sample_sl_batch(
                        rng_batch,
                        sl_state.sl_buffer,
                        config["sl_batch_size"],
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits, _ = network.apply(params, batch.obs)
                        logits_masked = jnp.where(batch.action_mask, logits, -jnp.inf)
                        log_probs = jax.nn.log_softmax(logits_masked, axis=-1)
                        action_log_probs = jnp.take_along_axis(
                            log_probs,
                            batch.action[:, None],
                            axis=-1,
                        ).squeeze(-1)
                        ce_loss = -action_log_probs.mean()
                        acc = (
                            jnp.argmax(logits_masked, axis=-1) == batch.action
                        ).mean()
                        return ce_loss, {
                            "sl_loss": ce_loss,
                            "sl_acc": acc,
                        }

                    grad_fn = jax.value_and_grad(sl_loss, has_aux=True)
                    (_, aux), grads = grad_fn(train_state.params)
                    aux["sl_grad_norm"] = pytree_norm(grads)
                    new_train_state = train_state.apply_gradients(grads=grads)
                    return new_train_state, aux

                def _skip_step(train_state: TrainState):
                    return train_state, {
                        "sl_loss": jnp.array(0.0, dtype=jnp.float32),
                        "sl_acc": jnp.array(0.0, dtype=jnp.float32),
                        "sl_grad_norm": jnp.array(0.0, dtype=jnp.float32),
                    }

                train_state, aux = lax.cond(
                    has_data,
                    _train_step,
                    _skip_step,
                    sl_state.train_state,
                )
                sl_state = SLUpdateState(
                    train_state=train_state,
                    sl_buffer=sl_state.sl_buffer,
                    rng=rng,
                )
                return sl_state, aux

            sl_state = SLUpdateState(
                train_state=avg_train_state,
                sl_buffer=sl_buffer,
                rng=rng_sl_update,
            )
            sl_state, sl_loss_info = lax.scan(
                update_sl_step,
                sl_state,
                None,
                config["sl_num_steps_per_update"],
            )
            avg_train_state = sl_state.train_state
            sl_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), sl_loss_info)

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
                "br_action_frac_rollout": transitions.is_br.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "update_step": runner_state.update_step,
            }
            metric.update(ppo_loss_info)
            metric.update(sl_loss_info)

            def logging_callback(seed_val, metric_dict, info, avg_params, br_params):
                avg_policy = policy_from_network(network.apply, avg_params)
                br_policy = policy_from_network(network.apply, br_params)
                avg_expl = exploitability(avg_policy)
                br_expl = exploitability(br_policy)
                metric_dict = dict(metric_dict)
                metric_dict["exploitability_avg_policy"] = float(avg_expl)
                metric_dict["exploitability_br_policy"] = float(br_expl)
                wandb_callback(seed_val, metric_dict, info)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                metric,
                transitions.info,
                avg_train_state.params,
                br_train_state.params,
            )

            runner_state = RunnerState(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                update_step=runner_state.update_step + 1,
                rng=rng,
            )
            return runner_state, None

        initial_runner_state = RunnerState(
            br_train_state=br_train_state,
            avg_train_state=avg_train_state,
            sl_buffer=sl_buffer,
            state=state,
            obs=obs,
            done=jnp.zeros((config["num_envs"]), dtype=jnp.bool_),
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )
        final_runner_state, _ = lax.scan(
            update_step,
            initial_runner_state,
            None,
            config["num_update_steps"],
        )
        return final_runner_state

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_ppo_nfsp")
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
