"""
PQN (Policy Gradient with Q-Network) + NFSP for heads-up limit Texas Hold'em.
"""

import datetime
from typing import Any, Callable, NamedTuple

import distrax
from flax import serialization
from flax.training.train_state import TrainState
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PRNGKeyArray,
)
from bluffjax import make
from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
    TexasNoLimitHoldEmState,
)
from bluffjax.networks.mlp import (
    ActorCriticDiscreteMLP,
    ActorDiscreteMLP,
    QNetworkDiscreteMLP,
)
from bluffjax.utils.jax_utils import pytree_norm
from bluffjax.utils.wandb_multilogger import WandbMultiLogger

# For REBEL value network comparison
from bluffjax.examples.HUNL.hunl_game_utils import PBS_INPUT_DIM
from bluffjax.examples.HUNL.hunl_rebel import (
    ValueNetworkMLP,
    _get_action_from_params,
)

LOGGER = None


class Transition(NamedTuple):
    obs: FloatArray
    action_mask: BoolArray
    action: IntArray
    reward: FloatArray  # reward for acting player (for TD)
    reward_full: FloatArray  # full reward per agent (for returns)
    absorbing: BoolArray
    done: BoolArray
    q_val: FloatArray
    player_idx: IntArray
    next_obs: FloatArray
    info: dict[str, Any]
    is_br: BoolArray


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
    state: TexasNoLimitHoldEmState
    obs: FloatArray
    done: BoolArray
    update_step: IntArray
    rng: PRNGKeyArray


class PQNUpdateState(NamedTuple):
    train_state: TrainState
    transitions: Transition
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
    env = make("texas_nolimit_holdem", **config["env_kwargs"])
    sample_state, _ = env.reset(jax.random.PRNGKey(0))
    action_dim = int(env.get_avail_actions(sample_state).shape[-1])
    config["batch_shuffle_dim"] = (
        config["num_steps_per_env_per_update"] * config["num_envs"]
    )

    compare_mode = config["compare_mode"]
    compare_against_random = compare_mode == "random"
    compare_enabled = config["compare"]
    compare_network_type = config["compare_network_type"]

    baseline_params = None
    rebel_value_network = None
    rebel_value_params = None
    actor_critic_network = None
    if compare_enabled and (not compare_against_random):
        checkpoint_path = config["compare_with"]
        if compare_network_type == "value":
            rebel_value_network = ValueNetworkMLP(
                hidden_dim=config["compare_value_hidden_dim"]
            )
            template_pbs = jnp.zeros((PBS_INPUT_DIM,), dtype=jnp.float32)
            template_params = rebel_value_network.init(
                jax.random.PRNGKey(0), template_pbs
            )
            with open(checkpoint_path, "rb") as f:
                rebel_value_params = serialization.from_bytes(template_params, f.read())

            def _squeeze_leading_batch(x):
                if hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == 1:
                    return jnp.squeeze(x, axis=0)
                return x

            rebel_value_params = jax.tree_util.tree_map(
                _squeeze_leading_batch, rebel_value_params
            )
            baseline_params = rebel_value_params
            print(
                f"Loaded REBEL value network baseline from {checkpoint_path} "
                f"(compare_network_type=value)"
            )
        elif compare_network_type == "actor_critic":
            actor_critic_network = ActorCriticDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            template_obs = jnp.zeros((env.obs_dim,), dtype=jnp.float32)
            template_params = actor_critic_network.init(
                jax.random.PRNGKey(0), template_obs
            )
            with open(checkpoint_path, "rb") as f:
                baseline_params = serialization.from_bytes(template_params, f.read())
            print(
                f"Loaded actor-critic baseline from {checkpoint_path} "
                f"(compare_network_type=actor_critic)"
            )
        else:
            raise ValueError(
                f"compare_network_type must be 'value' or 'actor_critic', "
                f"got '{compare_network_type}'"
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

    def init_sl_buffer(obs_dim: int, action_dim: int) -> SLBufferState:
        capacity = config["sl_reservoir_capacity"]
        return SLBufferState(
            obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
            action_mask=jnp.zeros((capacity, action_dim), dtype=jnp.bool_),
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
            buf, rng_inner = carry
            obs_s, action_mask_s, action_s, valid_s = sample
            rng_inner, rng_i = jax.random.split(rng_inner)

            def _add(cur: SLBufferState) -> SLBufferState:
                k = cur.seen
                j = jax.random.randint(rng_i, (), 0, k + 1, dtype=jnp.int32)
                not_full = k < capacity
                write_idx = jnp.where(not_full, k, j)
                should_write = not_full | (j < capacity)

                def _write(b: SLBufferState) -> SLBufferState:
                    return SLBufferState(
                        obs=b.obs.at[write_idx].set(obs_s),
                        action_mask=b.action_mask.at[write_idx].set(action_mask_s),
                        action=b.action.at[write_idx].set(action_s),
                        seen=b.seen,
                        size=b.size,
                    )

                cur = lax.cond(should_write, _write, lambda b: b, cur)
                return SLBufferState(
                    obs=cur.obs,
                    action_mask=cur.action_mask,
                    action=cur.action,
                    seen=cur.seen + 1,
                    size=jnp.minimum(capacity, cur.size + 1),
                )

            buf = lax.cond(valid_s, _add, lambda b: b, buf)
            return (buf, rng_inner), None

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

    def sample_action_actor(
        avg_net: ActorDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits = avg_net.apply(params, obs.astype(jnp.float32))
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_action_q_greedy(
        br_net: QNetworkDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        q_vals = br_net.apply(params, obs.astype(jnp.float32))
        q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
        best_val = jnp.max(q_vals_masked)
        ties = (q_vals_masked == best_val) & action_mask
        logits = jnp.where(ties, 0.0, -1e9)
        return jax.random.categorical(rng, logits)

    def sample_masked_action_actor_critic(
        ac_net: ActorCriticDiscreteMLP,
        params: Any,
        obs: FloatArray,
        action_mask: BoolArray,
        rng: PRNGKeyArray,
    ) -> IntArray:
        logits, _ = ac_net.apply(params, obs)
        logits_masked = jnp.where(action_mask, logits, -jnp.inf)
        return distrax.Categorical(logits=logits_masked).sample(seed=rng)

    def sample_random_legal_action(
        action_mask: BoolArray, rng: PRNGKeyArray
    ) -> IntArray:
        probs = action_mask.astype(jnp.float32)
        probs = probs / jnp.maximum(probs.sum(), 1.0)
        return distrax.Categorical(probs=probs).sample(seed=rng)

    def compare_single_episode(
        rng: PRNGKeyArray,
        train_params: Any,
        checkpoint_params: Any,
        train_player_idx: IntArray,
        use_avg_policy: bool,
        br_net: QNetworkDiscreteMLP,
        avg_net: ActorDiscreteMLP,
    ) -> tuple[PRNGKeyArray, FloatArray, FloatArray, FloatArray]:
        rng, rng_reset = jax.random.split(rng)
        env_state, obs = env.reset(rng_reset)

        def play_cond(carry):
            _, _, _, _, done_flag, _ = carry
            return ~done_flag

        def play_body(carry):
            state_s, obs_s, rng_s, step_count, _, terminal_rewards = carry
            rng_s, rng_train, rng_base, rng_step = jax.random.split(rng_s, 4)
            current_player = state_s.current_player_idx
            action_mask = env.get_avail_actions(state_s)

            if use_avg_policy:
                train_action = sample_action_actor(
                    avg_net, train_params, obs_s, action_mask, rng_train
                )
            else:
                train_action = sample_action_q_greedy(
                    br_net, train_params, obs_s, action_mask, rng_train
                )

            if compare_against_random:
                baseline_action = sample_random_legal_action(action_mask, rng_base)
            elif compare_network_type == "value":
                baseline_action, _ = _get_action_from_params(
                    env,
                    state_s,
                    rebel_value_network,
                    rebel_value_params,
                    max_depth=4,
                    rng=rng_base,
                )
            else:
                baseline_action = sample_masked_action_actor_critic(
                    actor_critic_network,
                    checkpoint_params,
                    obs_s,
                    action_mask,
                    rng_base,
                )
            action = jnp.where(
                current_player == train_player_idx, train_action, baseline_action
            )

            next_state, next_obs, reward, _, done, _ = env.step(
                rng_step, state_s, action
            )
            new_terminal_rewards = jnp.where(done, reward, terminal_rewards)
            return (
                next_state,
                next_obs,
                rng_s,
                step_count + 1,
                done,
                new_terminal_rewards,
            )

        _, _, rng, step_count, done_flag, terminal_rewards = lax.while_loop(
            play_cond,
            play_body,
            (
                env_state,
                obs,
                rng,
                jnp.array(0, dtype=jnp.int32),
                jnp.bool_(False),
                jnp.zeros(env.num_agents, dtype=jnp.float32),
            ),
        )
        train_chips = terminal_rewards[train_player_idx]
        return (
            rng,
            train_chips,
            step_count.astype(jnp.float32),
            done_flag.astype(jnp.float32),
        )

    def run_compare_eval(
        rng: PRNGKeyArray,
        avg_params: Any,
        br_params: Any,
        checkpoint_params: Any,
        br_net: QNetworkDiscreteMLP,
        avg_net: ActorDiscreteMLP,
    ) -> tuple[PRNGKeyArray, FloatArray, FloatArray, FloatArray]:
        def compare_episode(carry, episode_idx):
            rng_s, avg_p, br_p, base_p = carry
            swap = (episode_idx % 2) == 1
            avg_train_idx = jnp.where(swap, 1, 0)
            br_train_idx = jnp.where(swap, 1, 0)

            rng_s, avg_chips, avg_len, avg_done = compare_single_episode(
                rng_s,
                avg_p,
                base_p,
                avg_train_idx,
                use_avg_policy=True,
                br_net=br_net,
                avg_net=avg_net,
            )
            rng_s, br_chips, br_len, br_done = compare_single_episode(
                rng_s,
                br_p,
                base_p,
                br_train_idx,
                use_avg_policy=False,
                br_net=br_net,
                avg_net=avg_net,
            )
            return (rng_s, avg_p, br_p, base_p), (
                avg_chips,
                br_chips,
                avg_len,
                br_len,
                avg_done,
                br_done,
            )

        (rng, _, _, _), (
            avg_chips_arr,
            br_chips_arr,
            avg_len_arr,
            br_len_arr,
            avg_done_arr,
            br_done_arr,
        ) = lax.scan(
            compare_episode,
            (rng, avg_params, br_params, checkpoint_params),
            jnp.arange(config["compare_steps"]),
        )
        avg_done_count = jnp.maximum(avg_done_arr.sum(), 1.0)
        br_done_count = jnp.maximum(br_done_arr.sum(), 1.0)
        avg_chips = avg_chips_arr.sum() / avg_done_count
        br_chips = br_chips_arr.sum() / br_done_count
        avg_eval_len = (
            avg_len_arr.sum() / avg_done_count + br_len_arr.sum() / br_done_count
        ) / 2
        return rng, avg_chips, br_chips, avg_eval_len

    def train(rng: PRNGKeyArray, seed: int) -> RunnerState:
        def train_setup(
            rng_inner: PRNGKeyArray,
        ) -> tuple[
            TrainState,
            TrainState,
            QNetworkDiscreteMLP,
            ActorDiscreteMLP,
            TexasNoLimitHoldEmState,
            FloatArray,
            SLBufferState,
        ]:
            rng_inner, rng_reset = jax.random.split(rng_inner)
            rng_resets = jax.random.split(rng_reset, config["num_envs"])
            state, obs = jax.vmap(env.reset, in_axes=(0))(rng_resets)

            rng_inner, rng_br_init, rng_avg_init = jax.random.split(rng_inner, 3)
            br_network = QNetworkDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            avg_network = ActorDiscreteMLP(
                action_dim=action_dim, hidden_dim=config["fc_dim_size"]
            )
            br_params = br_network.init(rng_br_init, obs)
            avg_params = avg_network.init(rng_avg_init, obs)

            br_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.radam(learning_rate=linear_decay),
            )
            avg_tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(learning_rate=config["sl_lr"], eps=1e-5),
            )

            br_train_state = TrainState.create(
                apply_fn=br_network.apply, params=br_params, tx=br_tx
            )
            avg_train_state = TrainState.create(
                apply_fn=avg_network.apply, params=avg_params, tx=avg_tx
            )
            sl_buffer = init_sl_buffer(obs.shape[-1], action_dim)
            return (
                br_train_state,
                avg_train_state,
                br_network,
                avg_network,
                state,
                obs,
                sl_buffer,
            )

        rng, rng_setup = jax.random.split(rng)
        (
            br_train_state,
            avg_train_state,
            br_network,
            avg_network,
            state,
            obs,
            sl_buffer,
        ) = train_setup(rng_setup)

        def update_step(
            runner_state: RunnerState,
            unused: None,
        ) -> tuple[RunnerState, None]:
            def step(
                runner_state_inner: RunnerState,
                unused: None,
            ) -> tuple[RunnerState, Transition]:
                br_train_state_s = runner_state_inner.br_train_state
                avg_train_state_s = runner_state_inner.avg_train_state
                state_s = runner_state_inner.state
                obs_s = runner_state_inner.obs
                rng_s = runner_state_inner.rng

                rng_s, rng_mix, rng_action, rng_step = jax.random.split(rng_s, 4)
                action_mask = jax.vmap(env.get_avail_actions, in_axes=(0))(state_s)

                q_vals = br_network.apply(
                    br_train_state_s.params, obs_s.astype(jnp.float32)
                )
                q_vals_masked = jnp.where(action_mask, q_vals, -jnp.inf)
                greedy_action = jnp.argmax(q_vals_masked, axis=-1)

                eps = epsilon_schedule(
                    jnp.array(runner_state_inner.update_step, dtype=jnp.float32)
                )
                rng_uniform, rng_random = jax.random.split(rng_action, 2)
                explore = jax.random.uniform(rng_uniform, (config["num_envs"],)) < eps
                random_probs = action_mask.astype(jnp.float32) / (
                    action_mask.sum(axis=-1, keepdims=True) + 1e-8
                )
                rng_randoms = jax.random.split(rng_random, config["num_envs"])
                random_action = jax.vmap(
                    lambda rng, p: distrax.Categorical(probs=p).sample(seed=rng)
                )(rng_randoms, random_probs)
                br_action = jnp.where(explore, random_action, greedy_action)

                avg_logits = avg_network.apply(
                    avg_train_state_s.params, obs_s.astype(jnp.float32)
                )
                avg_logits_masked = jnp.where(action_mask, avg_logits, -jnp.inf)
                avg_pi = distrax.Categorical(logits=avg_logits_masked)
                rng_avgs = jax.random.split(rng_action, config["num_envs"])
                avg_action = jax.vmap(lambda pi, r: pi.sample(seed=r))(avg_pi, rng_avgs)

                is_br = jax.random.bernoulli(
                    rng_mix,
                    p=config["anticipatory_eta"],
                    shape=(config["num_envs"],),
                )
                action = jnp.where(is_br, br_action, avg_action)

                chosen_q = jnp.take_along_axis(
                    q_vals, action[..., None].astype(jnp.int32), axis=-1
                ).squeeze(-1)

                rng_steps = jax.random.split(rng_step, config["num_envs"])
                next_state, next_obs, reward, absorbing, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_steps, state_s, action)

                reward_acting = jnp.take_along_axis(
                    reward,
                    state_s.current_player_idx[..., None].astype(jnp.int32),
                    axis=-1,
                ).squeeze(-1)

                transition = Transition(
                    obs=obs_s,
                    action_mask=action_mask,
                    action=action,
                    reward=reward_acting,
                    reward_full=reward,
                    absorbing=absorbing,
                    done=done,
                    q_val=chosen_q,
                    player_idx=state_s.current_player_idx,
                    next_obs=next_obs,
                    info=info,
                    is_br=is_br,
                )
                runner_state_inner = RunnerState(
                    br_train_state=br_train_state_s,
                    avg_train_state=avg_train_state_s,
                    sl_buffer=runner_state_inner.sl_buffer,
                    state=next_state,
                    obs=next_obs,
                    done=done,
                    update_step=runner_state_inner.update_step,
                    rng=rng_s,
                )
                return runner_state_inner, transition

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
            rng, rng_sl_append, rng_pqn_update, rng_sl_update, rng_compare = (
                jax.random.split(runner_state.rng, 5)
            )

            last_q_raw = jnp.max(
                br_network.apply(br_train_state.params, last_obs.astype(jnp.float32)),
                axis=-1,
            )
            last_q = -last_q_raw

            def compute_q_lambda_targets(
                transitions: Transition,
                last_q: FloatArray,
                br_params,
            ) -> FloatArray:
                gamma = jnp.float32(config["gamma"])
                q_lambda = jnp.float32(config["q_lambda"])
                reward = transitions.reward
                next_obs = transitions.next_obs.astype(jnp.float32)
                done = transitions.done
                q_vals_next = br_network.apply(br_params, next_obs)

                done_f32 = done.astype(jnp.float32)
                lambda_returns_last = reward[-1] + gamma * (1.0 - done_f32[-1]) * last_q
                idx = jnp.maximum(0, q_vals_next.shape[0] - 2)
                last_max_q = -jnp.max(q_vals_next[idx], axis=-1)

                def _get_target(carry, rew_q_done):
                    reward_t, q_next_t, done_t = rew_q_done
                    lambda_returns, next_q = carry
                    done_t_f32 = done_t.astype(jnp.float32)
                    target_bootstrap = reward_t + gamma * (1.0 - done_t_f32) * next_q
                    delta = -lambda_returns - next_q
                    lambda_returns_new = target_bootstrap + gamma * q_lambda * delta
                    lambda_returns_new = (
                        1.0 - done_t_f32
                    ) * lambda_returns_new + done_t_f32 * reward_t
                    next_q_new = -jnp.max(q_next_t, axis=-1)
                    return (lambda_returns_new, next_q_new), lambda_returns_new

                init_carry = (lambda_returns_last, last_max_q)
                reward_prev = reward[:-1]
                q_next_prev = q_vals_next[:-1]
                done_prev = done_f32[:-1]
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

            targets = compute_q_lambda_targets(
                transitions, last_q, br_train_state.params
            )

            obs_flat = transitions.obs.reshape(-1, transitions.obs.shape[-1])
            action_mask_flat = transitions.action_mask.reshape(
                -1, transitions.action_mask.shape[-1]
            )
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

            def _reshape_batch(x):
                if x.ndim == 2:
                    return x.reshape(-1)
                elif x.ndim == 3:
                    return x.reshape(-1, *x.shape[2:])
                else:
                    return x.reshape(-1, *x.shape[3:])

            def _reshape_transition(trans: Transition) -> Transition:
                info_reshaped = {k: _reshape_batch(v) for k, v in trans.info.items()}
                return Transition(
                    obs=_reshape_batch(trans.obs),
                    action_mask=_reshape_batch(trans.action_mask),
                    action=_reshape_batch(trans.action),
                    reward=_reshape_batch(trans.reward),
                    reward_full=_reshape_batch(trans.reward_full),
                    absorbing=_reshape_batch(trans.absorbing),
                    done=_reshape_batch(trans.done),
                    q_val=_reshape_batch(trans.q_val),
                    player_idx=_reshape_batch(trans.player_idx),
                    next_obs=_reshape_batch(trans.next_obs),
                    info=info_reshaped,
                    is_br=_reshape_batch(trans.is_br),
                )

            def update_epoch(
                pqn_state: PQNUpdateState,
                unused: None,
            ) -> tuple[PQNUpdateState, dict[str, FloatArray]]:
                rng_s, rng_permute = jax.random.split(pqn_state.rng)
                trans_reshaped = _reshape_transition(pqn_state.transitions)
                targets_reshaped = _reshape_batch(pqn_state.targets)

                permutation = jax.random.permutation(
                    rng_permute, config["batch_shuffle_dim"]
                )
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
                    reward_full=jnp.take(
                        trans_reshaped.reward_full, permutation, axis=0
                    ),
                    absorbing=jnp.take(trans_reshaped.absorbing, permutation, axis=0),
                    done=jnp.take(trans_reshaped.done, permutation, axis=0),
                    q_val=jnp.take(trans_reshaped.q_val, permutation, axis=0),
                    player_idx=jnp.take(trans_reshaped.player_idx, permutation, axis=0),
                    next_obs=jnp.take(trans_reshaped.next_obs, permutation, axis=0),
                    info=info_shuffled,
                    is_br=jnp.take(trans_reshaped.is_br, permutation, axis=0),
                )
                targets_shuffled = jnp.take(targets_reshaped, permutation, axis=0)

                batch_size = config["batch_shuffle_dim"]
                minibatch_size = batch_size // config["num_minibatches"]
                info_minibatches = {
                    k: (
                        v.reshape(
                            config["num_minibatches"],
                            minibatch_size,
                            *v.shape[1:],
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
                    reward_full=trans_shuffled.reward_full.reshape(
                        config["num_minibatches"], minibatch_size, -1
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
                    is_br=trans_shuffled.is_br.reshape(
                        config["num_minibatches"], minibatch_size
                    ),
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
                        q_vals = br_network.apply(params, trans.obs.astype(jnp.float32))
                        chosen_q = jnp.take_along_axis(
                            q_vals,
                            trans.action[..., None].astype(jnp.int32),
                            axis=-1,
                        ).squeeze(-1)
                        mask = trans.is_br.astype(jnp.float32)
                        mask_denom = jnp.maximum(mask.sum(), 1.0)
                        td_loss = (
                            0.5 * jnp.square(chosen_q - targets_mb) * mask
                        ).sum() / mask_denom
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
                    pqn_state.train_state,
                    minibatches,
                )
                pqn_state = PQNUpdateState(
                    train_state=final_train_state,
                    transitions=pqn_state.transitions,
                    targets=pqn_state.targets,
                    rng=rng_s,
                )
                return pqn_state, batch_stats

            pqn_state = PQNUpdateState(
                train_state=br_train_state,
                transitions=transitions,
                targets=targets,
                rng=rng_pqn_update,
            )
            final_pqn_state, pqn_loss_info = lax.scan(
                update_epoch,
                pqn_state,
                None,
                config["num_epochs"],
            )
            br_train_state = final_pqn_state.train_state
            pqn_loss_info = jax.tree_util.tree_map(lambda x: x.mean(), pqn_loss_info)

            def update_sl_step(
                sl_state: SLUpdateState,
                unused: None,
            ) -> tuple[SLUpdateState, dict[str, FloatArray]]:
                rng_s, rng_batch = jax.random.split(sl_state.rng)
                has_data = sl_state.sl_buffer.size > 0

                def _train_step(train_state: TrainState):
                    batch = sample_sl_batch(
                        rng_batch,
                        sl_state.sl_buffer,
                        config["sl_batch_size"],
                    )

                    def sl_loss(params) -> tuple[FloatArray, dict[str, FloatArray]]:
                        logits = avg_network.apply(
                            params, batch.obs.astype(jnp.float32)
                        )
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
                    rng=rng_s,
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
                transitions.done[..., None], transitions.reward_full, 0.0
            )
            done_count = jnp.maximum(transitions.done.sum(), 1)
            returns_avg = masked_returns.sum() / done_count
            returns_avg_agent_one = masked_returns[..., 0].sum() / done_count
            returns_avg_agent_two = masked_returns[..., 1].sum() / done_count

            masked_timesteps = jnp.where(
                transitions.done,
                transitions.info["timestep"].astype(jnp.float32),
                0.0,
            )
            ep_length_avg = masked_timesteps.sum() / done_count

            should_compare = compare_enabled & (
                runner_state.update_step % config["compare_interval"] == 0
            )

            def do_compare(carry_rng):
                checkpoint_params = baseline_params
                if compare_against_random:
                    checkpoint_params = avg_train_state.params
                return run_compare_eval(
                    carry_rng,
                    avg_train_state.params,
                    br_train_state.params,
                    checkpoint_params,
                    br_network,
                    avg_network,
                )

            def skip_compare(carry_rng):
                return (
                    carry_rng,
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                )

            (
                rng_compare,
                avg_chips_avg_vs_baseline,
                avg_chips_br_vs_baseline,
                avg_eval_episode_length,
            ) = lax.cond(
                should_compare,
                do_compare,
                skip_compare,
                rng_compare,
            )

            metric = {
                "returns_avg": returns_avg,
                "returns_avg_agent_one": returns_avg_agent_one,
                "returns_avg_agent_two": returns_avg_agent_two,
                "ep_length_avg": ep_length_avg,
                "br_action_frac_rollout": transitions.is_br.mean(),
                "sl_buffer_size": sl_buffer.size.astype(jnp.float32),
                "sl_buffer_seen": sl_buffer.seen.astype(jnp.float32),
                "update_step": runner_state.update_step,
                "avg_chips_avg_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_chips_avg_vs_baseline
                ),
                "avg_chips_br_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_chips_br_vs_baseline
                ),
                "avg_chips_avg_vs_random": (
                    avg_chips_avg_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_chips_br_vs_random": (
                    avg_chips_br_vs_baseline
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_eval_episode_length": avg_eval_episode_length,
            }
            metric.update(pqn_loss_info)
            metric.update(sl_loss_info)

            def logging_callback(seed_val, metric_dict, info):
                wandb_callback(seed_val, dict(metric_dict), info)

            jax.experimental.io_callback(
                logging_callback,
                None,
                seed,
                metric,
                transitions.info,
            )

            runner_state = RunnerState(
                br_train_state=br_train_state,
                avg_train_state=avg_train_state,
                sl_buffer=sl_buffer,
                state=runner_state.state,
                obs=runner_state.obs,
                done=runner_state.done,
                update_step=runner_state.update_step + 1,
                rng=rng_compare,
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


@hydra.main(version_base=None, config_path="./", config_name="config_pqn_nfsp")
def main(config: dict) -> None:
    global LOGGER
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
        if LOGGER is not None:
            LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
