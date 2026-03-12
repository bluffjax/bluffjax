"""
ReBeL (Recursive Belief-based Learning) for heads-up limit Texas hold'em.

Full pipeline: depth-limited subgames, CFR-D, value network, self-play.
Training loop is fully jittable.
"""

import datetime
import os
from typing import Callable, NamedTuple

import flax.linen as nn
from flax import serialization
import hydra
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
import optax

from bluffjax.utils.typing import FloatArray, IntArray, PRNGKeyArray
from bluffjax import make
from bluffjax.environments.texas_limit_holdem.texas_limit_holdem import (
    TexasLimitHoldEmState,
    TexasLimitHoldem,
)
from bluffjax.examples.HUL.hul_game_utils import (
    NUM_ACTIONS,
    NUM_BUCKETS,
    PBS_INPUT_DIM,
    VALUE_OUTPUT_DIM,
    encode_pbs,
    pbs_from_state,
)
from bluffjax.utils.game_utils.poker_utils import _compare_hands
from bluffjax.utils.wandb_multilogger import WandbMultiLogger


# =============================================================================
# Constants
# =============================================================================

MAX_INFOSETS = 8192
HISTORY_BASE = 1 + NUM_ACTIONS + NUM_ACTIONS**2 + NUM_ACTIONS**3


# =============================================================================
# Value network
# =============================================================================


class ValueNetworkMLP(nn.Module):
    """MLP that maps PBS encoding to values per hand bucket (NUM_BUCKETS per player)."""

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: FloatArray) -> FloatArray:
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            VALUE_OUTPUT_DIM,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return x


# =============================================================================
# History encoding and infoset indexing
# =============================================================================


def _child_hist_enc(hist_enc: IntArray, action: IntArray) -> IntArray:
    """Child history_enc when taking action from hist_enc."""
    d1_end = NUM_ACTIONS
    d2_start = 1 + NUM_ACTIONS
    d2_end = d2_start + NUM_ACTIONS**2 - 1
    d3_start = d2_start + NUM_ACTIONS**2
    return jnp.where(
        hist_enc == 0,
        1 + action,
        jnp.where(
            hist_enc <= d1_end,
            d2_start + (hist_enc - 1) + NUM_ACTIONS * action,
            jnp.where(
                hist_enc <= d2_end,
                d3_start + (hist_enc - d2_start) + (NUM_ACTIONS**2) * action,
                -1,
            ),
        ),
    )


def infoset_idx(player: IntArray, bucket: IntArray, hist_enc: IntArray) -> IntArray:
    """Compute infoset index: (player*NUM_BUCKETS + bucket) * 85 + hist_enc."""
    return (player * NUM_BUCKETS + bucket) * HISTORY_BASE + hist_enc


# =============================================================================
# Tree structure helpers (for multi-depth CFR)
# =============================================================================


def _node_count(max_depth: int) -> int:
    """Number of nodes in depth-limited k-ary tree."""
    return (NUM_ACTIONS ** (max_depth + 1) - 1) // (NUM_ACTIONS - 1)


def _depth_start(depth: int) -> int:
    """First node index at given depth for k-ary tree."""
    return (NUM_ACTIONS**depth - 1) // (NUM_ACTIONS - 1)


def _depth_and_path(node_idx: IntArray, max_depth: int) -> tuple[IntArray, IntArray]:
    """Map flat node index to (depth, path_idx). path_idx in [0, 4^depth)."""
    depth_starts = jnp.array(
        [_depth_start(d) for d in range(max_depth + 2)], dtype=jnp.int32
    )
    ge = (node_idx >= depth_starts[:-1]).astype(jnp.int32)
    depth = jnp.sum(ge) - 1
    depth = jnp.clip(depth, 0, max_depth)
    path_idx = node_idx - jnp.take(depth_starts, depth)
    return depth, path_idx


def _path_to_actions(path_idx: IntArray, depth: IntArray, max_depth: int) -> IntArray:
    """Decode path_idx to (max_depth,) array of actions in base NUM_ACTIONS."""
    powers = jnp.array([NUM_ACTIONS**i for i in range(max_depth)], dtype=jnp.int32)

    def get_action(i):
        return (path_idx // powers[i]) % NUM_ACTIONS

    actions = jax.vmap(get_action)(jnp.arange(max_depth))
    valid = jnp.arange(max_depth) < depth
    return jnp.where(valid, actions, jnp.zeros(max_depth, dtype=jnp.int32))


class PathApplyResult(NamedTuple):
    state: TexasLimitHoldEmState
    path_valid: IntArray
    terminated: IntArray
    terminal_rewards: FloatArray


def _apply_path(
    env: TexasLimitHoldem,
    root_state: TexasLimitHoldEmState,
    path_actions: IntArray,
    depth: IntArray,
) -> PathApplyResult:
    """Apply path from root via lax.scan using env.step auto-reset semantics."""
    step_key = jax.random.PRNGKey(0)

    def body(carry, i):
        state, path_actions_arr, d, path_valid, terminated, terminal_rewards = carry
        step_valid = (i < d).astype(jnp.bool_)
        action = path_actions_arr[i]
        avail = env.get_avail_actions(state)
        action_legal = jnp.take(avail, action)
        step_ok = step_valid & action_legal & (~terminated)
        # If path tries to continue after terminal, mark invalid.
        continued_after_terminal = step_valid & terminated
        new_valid = path_valid & (~continued_after_terminal) & (step_ok | (~step_valid))

        def do_step():
            next_state, _, rewards, _, done, _ = env.step(step_key, state, action)
            new_terminal_rewards = jnp.where(done, rewards, terminal_rewards)
            return next_state, done, new_terminal_rewards

        def skip_step():
            return state, terminated, terminal_rewards

        next_state, new_terminated, new_terminal_rewards = lax.cond(
            step_ok, do_step, skip_step
        )
        return (
            next_state,
            path_actions_arr,
            d,
            new_valid,
            new_terminated,
            new_terminal_rewards,
        ), ()

    (final_state, _, _, path_valid, terminated, terminal_rewards), _ = lax.scan(
        body,
        (
            root_state,
            path_actions,
            depth,
            jnp.bool_(True),
            jnp.bool_(False),
            jnp.zeros(2, dtype=jnp.float32),
        ),
        jnp.arange(path_actions.shape[0]),
    )
    return PathApplyResult(
        state=final_state,
        path_valid=path_valid,
        terminated=terminated,
        terminal_rewards=terminal_rewards,
    )


def _hist_enc_from_path(path_actions: IntArray, depth: IntArray) -> IntArray:
    """Compute history encoding from path. Iteratively apply _child_hist_enc."""

    def step_hist(hist_enc, action_and_valid):
        action, valid = action_and_valid
        new_enc = _child_hist_enc(hist_enc, action)
        return jnp.where(valid, new_enc, hist_enc), ()

    max_d = path_actions.shape[0]
    valid_mask = jnp.arange(max_d) < depth

    hist_enc, _ = lax.scan(
        step_hist,
        jnp.int32(0),
        (path_actions, valid_mask),
    )
    return hist_enc


# =============================================================================
# Regret matching (JAX)
# =============================================================================


def regret_matching(regrets: FloatArray, legal_mask: FloatArray) -> FloatArray:
    """Regret matching: strategy proportional to positive regrets."""
    pos_regrets = jnp.maximum(regrets, 0.0)
    pos_regrets = jnp.where(legal_mask, pos_regrets, 0.0)
    total = jnp.sum(pos_regrets)
    probs = jnp.where(
        total > 1e-8,
        pos_regrets / total,
        jnp.where(legal_mask, 1.0 / (jnp.sum(legal_mask) + 1e-8), 0.0),
    )
    return probs


# =============================================================================
# Subgame tree and CFR (recursive, builds on the fly)
# =============================================================================


def _cfr_traversal_multi_depth(
    env: TexasLimitHoldem,
    root_state: TexasLimitHoldEmState,
    deal: tuple[IntArray, IntArray],
    cum_reg: FloatArray,
    cum_pol: FloatArray,
    value_fn: Callable[[FloatArray], FloatArray],
    iteration: IntArray,
    max_depth: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    CFR traversal for configurable max_depth. Builds flat tree, computes values
    in reverse depth order, updates regrets.
    """
    node_count = _node_count(max_depth)
    depth_starts = jnp.array(
        [_depth_start(d) for d in range(max_depth + 2)], dtype=jnp.int32
    )

    def reverse_scan_body(carry, node_i):
        values, cr, cp = carry
        depth, path_idx = _depth_and_path(node_i, max_depth)
        path_actions = _path_to_actions(path_idx, depth, max_depth)
        path_result = _apply_path(env, root_state, path_actions, depth)
        state = path_result.state
        path_valid = path_result.path_valid
        terminated = path_result.terminated
        terminal_rewards = path_result.terminal_rewards
        avail = env.get_avail_actions(state)
        is_terminal = terminated
        num_legal = jnp.sum(avail.astype(jnp.float32))
        is_terminal = is_terminal | (num_legal < 1e-6)
        node_valid = path_valid

        b0, b1 = deal
        cur = state.current_player_idx
        bucket = jnp.where(cur == 0, b0, b1)
        hist_enc = _hist_enc_from_path(path_actions, depth)
        infoset = jnp.minimum(infoset_idx(cur, bucket, hist_enc), MAX_INFOSETS - 1)
        legal_mask = avail.astype(jnp.float32)

        def terminal_val():
            return jnp.where(terminated, terminal_rewards, jnp.zeros(2))

        def leaf_val():
            pbs = pbs_from_state(state, NUM_BUCKETS)
            enc = encode_pbs(pbs, NUM_BUCKETS)
            vals = value_fn(enc)
            v0 = jnp.mean(vals[:NUM_BUCKETS])
            v1 = jnp.mean(vals[NUM_BUCKETS:])
            return jnp.array([v0, v1])

        def internal_val():
            reg = cr[infoset]
            policy = regret_matching(reg, legal_mask)
            next_depth_start = jnp.take(depth_starts, depth + 1)
            child_indices = (
                next_depth_start + path_idx * NUM_ACTIONS + jnp.arange(NUM_ACTIONS)
            )
            cv0 = jnp.where(
                avail,
                jnp.take(values[:, 0], child_indices),
                jnp.zeros(NUM_ACTIONS),
            )
            cv1 = jnp.where(
                avail,
                jnp.take(values[:, 1], child_indices),
                jnp.zeros(NUM_ACTIONS),
            )
            v0_node = jnp.sum(policy * cv0)
            v1_node = jnp.sum(policy * cv1)
            return jnp.array([v0_node, v1_node])

        val = lax.cond(
            is_terminal,
            terminal_val,
            lambda: lax.cond(
                depth == max_depth,
                leaf_val,
                internal_val,
            ),
        )
        val = jnp.where(node_valid, val, jnp.zeros(2))
        values = values.at[node_i, :].set(val)

        def do_regret_update():
            reg = cr[infoset]
            policy = regret_matching(reg, legal_mask)
            next_depth_start = jnp.take(depth_starts, depth + 1)
            child_indices = (
                next_depth_start + path_idx * NUM_ACTIONS + jnp.arange(NUM_ACTIONS)
            )
            cv0 = jnp.where(avail, jnp.take(values[:, 0], child_indices), 0.0)
            cv1 = jnp.where(avail, jnp.take(values[:, 1], child_indices), 0.0)
            v0_node = jnp.sum(policy * cv0)
            v1_node = jnp.sum(policy * cv1)
            cf_reach = 1.0
            regret_updates = jnp.where(
                legal_mask > 0.5,
                jnp.where(
                    cur == 0,
                    cf_reach * (cv0 - v0_node),
                    cf_reach * (cv1 - v1_node),
                ),
                0.0,
            )
            new_cr = cr.at[infoset].add(regret_updates)
            w = iteration.astype(jnp.float32)
            new_cp = cp.at[infoset].add(w * policy)
            return new_cr, new_cp

        def no_regret_update():
            return cr, cp

        cr, cp = lax.cond(
            node_valid & (~is_terminal) & (depth < max_depth),
            do_regret_update,
            no_regret_update,
        )

        return (values, cr, cp), ()

    values = jnp.zeros((node_count, 2), dtype=jnp.float32)
    reverse_order = jnp.arange(node_count - 1, -1, -1)
    (values, cum_reg, cum_pol), _ = lax.scan(
        reverse_scan_body,
        (values, cum_reg, cum_pol),
        reverse_order,
    )

    v0_root = values[0, 0]
    v1_root = values[0, 1]
    return v0_root, v1_root, cum_reg, cum_pol


def run_cfr_on_tree(
    env: TexasLimitHoldem,
    root_state: TexasLimitHoldEmState,
    value_fn: Callable[[FloatArray], FloatArray],
    num_iterations: int,
    max_depth: int,
    rng: PRNGKeyArray,
) -> tuple[FloatArray, FloatArray]:
    """Run CFR on subgame. Returns (root_values_avg, final_cum_reg)."""
    cum_reg = jnp.zeros((MAX_INFOSETS, NUM_ACTIONS), dtype=jnp.float32)
    cum_pol = jnp.zeros((MAX_INFOSETS, NUM_ACTIONS), dtype=jnp.float32)
    root_vals_sum = jnp.zeros(2, dtype=jnp.float32)

    def cfr_body(carry, t):
        cum_reg, cum_pol, rv_sum, rng_key = carry
        rng_key, rng_b0, rng_b1 = jax.random.split(rng_key, 3)
        b0 = jax.random.randint(rng_b0, (), 0, NUM_BUCKETS)
        b1 = jax.random.randint(rng_b1, (), 0, NUM_BUCKETS)
        deal = (b0, b1)
        v0, v1, new_reg, new_pol = _cfr_traversal_multi_depth(
            env,
            root_state,
            deal,
            cum_reg,
            cum_pol,
            value_fn,
            jnp.int32(t + 1),
            max_depth,
        )
        rv_sum = rv_sum + jnp.array([v0, v1])
        return (new_reg, new_pol, rv_sum, rng_key), ()

    (cum_reg, cum_pol, root_vals_sum, _), _ = lax.scan(
        cfr_body,
        (cum_reg, cum_pol, root_vals_sum, rng),
        jnp.arange(num_iterations),
    )
    root_vals_avg = root_vals_sum / num_iterations
    return root_vals_avg, cum_reg


# =============================================================================
# Self-play and training
# =============================================================================


class ReplayBufferState(NamedTuple):
    pbs: FloatArray
    values: FloatArray
    seen: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    value_train_state: TrainState
    replay_buffer: ReplayBufferState
    env_state: TexasLimitHoldEmState
    rng: PRNGKeyArray
    update_step: IntArray


def _get_action_from_model(
    env: TexasLimitHoldem,
    state: TexasLimitHoldEmState,
    value_fn: Callable[[FloatArray], FloatArray],
    max_depth: int,
    rng: PRNGKeyArray,
) -> tuple[IntArray, PRNGKeyArray]:
    """Get action via argmax over value network outputs at each child."""
    avail = env.get_avail_actions(state)
    cur = state.current_player_idx

    def action_value(a):
        next_state, _, rewards, _, done, _ = env.step(jax.random.PRNGKey(0), state, a)
        val = lax.cond(
            done,
            lambda: rewards[cur],
            lambda: _value_at_state(next_state, value_fn, cur),
        )
        return val

    action_values = jax.vmap(action_value)(jnp.arange(NUM_ACTIONS))
    action_values = jnp.where(avail, action_values, -jnp.inf)
    best_val = jnp.max(action_values)
    ties = (action_values == best_val) & avail
    logits = jnp.where(ties, 0.0, -1e9)
    rng, rng_choice = jax.random.split(rng)
    action = jax.random.categorical(rng_choice, logits)
    return action, rng


def _get_action_from_params(
    env: TexasLimitHoldem,
    state: TexasLimitHoldEmState,
    network: nn.Module,
    params: dict,
    max_depth: int,
    rng: PRNGKeyArray,
) -> tuple[IntArray, PRNGKeyArray]:
    """Get action via argmax, using network.apply(params, enc) for values."""
    avail = env.get_avail_actions(state)
    cur = state.current_player_idx

    def action_value(a):
        next_state, _, rewards, _, done, _ = env.step(jax.random.PRNGKey(0), state, a)
        val = lax.cond(
            done,
            lambda: rewards[cur],
            lambda: _value_at_state_params(next_state, network, params, cur),
        )
        return val

    action_values = jax.vmap(action_value)(jnp.arange(NUM_ACTIONS))
    action_values = jnp.where(avail, action_values, -jnp.inf)
    best_val = jnp.max(action_values)
    ties = (action_values == best_val) & avail
    logits = jnp.where(ties, 0.0, -1e9)
    rng, rng_choice = jax.random.split(rng)
    action = jax.random.categorical(rng_choice, logits)
    return action, rng


def _value_at_state_params(
    state: TexasLimitHoldEmState,
    network: nn.Module,
    params: dict,
    cur: IntArray,
) -> FloatArray:
    pbs = pbs_from_state(state, NUM_BUCKETS)
    enc = encode_pbs(pbs, NUM_BUCKETS)
    vals = network.apply(params, enc)
    return jnp.where(
        cur == 0,
        jnp.mean(vals[:NUM_BUCKETS]),
        jnp.mean(vals[NUM_BUCKETS:]),
    )


def _value_at_state(
    state: TexasLimitHoldEmState,
    value_fn: Callable[[FloatArray], FloatArray],
    cur: IntArray,
) -> FloatArray:
    pbs = pbs_from_state(state, NUM_BUCKETS)
    enc = encode_pbs(pbs, NUM_BUCKETS)
    vals = value_fn(enc)
    return jnp.where(
        cur == 0,
        jnp.mean(vals[:NUM_BUCKETS]),
        jnp.mean(vals[NUM_BUCKETS:]),
    )


def huber_loss(x: FloatArray, delta: float = 1.0) -> FloatArray:
    """Huber loss for value network."""
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x * x,
        delta * (abs_x - 0.5 * delta),
    )


LOGGER: WandbMultiLogger | None = None


def make_train(config: dict) -> Callable[[PRNGKeyArray, IntArray], dict]:
    """Create training function that returns final state and saves model."""

    def init_replay_buffer() -> ReplayBufferState:
        return ReplayBufferState(
            pbs=jnp.zeros(
                (config["replay_capacity"], PBS_INPUT_DIM), dtype=jnp.float32
            ),
            values=jnp.zeros(
                (config["replay_capacity"], VALUE_OUTPUT_DIM), dtype=jnp.float32
            ),
            seen=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def append_one(
        buffer: ReplayBufferState,
        pbs_s: FloatArray,
        values_s: FloatArray,
        rng: PRNGKeyArray,
    ) -> tuple[ReplayBufferState, PRNGKeyArray]:
        rng, rng_i = jax.random.split(rng)
        k = buffer.seen
        j = jax.random.randint(rng_i, (), 0, k + 1, dtype=jnp.int32)
        not_full = k < config["replay_capacity"]
        write_idx = jnp.where(not_full, k, j)
        should_write = not_full | (j < config["replay_capacity"])
        new_pbs = jnp.where(
            should_write, buffer.pbs.at[write_idx].set(pbs_s), buffer.pbs
        )
        new_values = jnp.where(
            should_write, buffer.values.at[write_idx].set(values_s), buffer.values
        )
        return (
            ReplayBufferState(
                pbs=new_pbs,
                values=new_values,
                seen=buffer.seen + 1,
                size=jnp.minimum(config["replay_capacity"], buffer.size + 1),
            ),
            rng,
        )

    def sample_batch(
        rng: PRNGKeyArray,
        buffer: ReplayBufferState,
        batch_size_val: int,
    ) -> tuple[FloatArray, FloatArray]:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(
            rng, (batch_size_val,), 0, max_size, dtype=jnp.int32
        )
        return buffer.pbs[indices], buffer.values[indices]

    def train(rng: PRNGKeyArray, exp_id: IntArray) -> dict:
        rng, rng_init = jax.random.split(rng)
        network = ValueNetworkMLP(hidden_dim=config["value_hidden_dim"])
        dummy = jnp.zeros(PBS_INPUT_DIM, dtype=jnp.float32)
        params = network.init(rng_init, dummy)
        compare_mode = config["compare_mode"]
        if compare_mode not in ("checkpoint", "random"):
            raise ValueError(
                "compare_mode must be either 'checkpoint' or 'random', "
                f"got '{compare_mode}'"
            )
        compare_against_random = compare_mode == "random"

        tx = optax.adam(learning_rate=config["value_lr"])
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        baseline_params = params
        if config["compare"] and (not compare_against_random):
            if os.path.exists(config["compare_with"]):
                with open(config["compare_with"], "rb") as f:
                    baseline_params = serialization.from_bytes(params, f.read())
                print(f"Loaded baseline model from {config['compare_with']}")
            else:
                print(
                    "Checkpoint comparison requested but checkpoint file was not found: "
                    f"{config['compare_with']}"
                )

        env = make("texas_limit_holdem", num_agents=2)
        replay_buffer = init_replay_buffer()
        rng, rng_reset = jax.random.split(rng)
        env_state, _ = env.reset(rng_reset)

        def hand_step(carry, _):
            env_st, buf, ts, rng_key, hand_done = carry
            value_fn = lambda enc: network.apply(ts.params, enc)
            avail = env.get_avail_actions(env_st)
            has_children = jnp.any(avail)

            pbs = pbs_from_state(env_st, NUM_BUCKETS)
            pbs_enc = encode_pbs(pbs, NUM_BUCKETS)
            rng_new, rng_cfr, rng_act, rng_step = jax.random.split(rng_key, 4)
            root_vals, _ = run_cfr_on_tree(
                env,
                env_st,
                value_fn,
                config["cfr_iterations"],
                config["max_depth"],
                rng_cfr,
            )
            values_target = jnp.concatenate(
                [
                    jnp.full(NUM_BUCKETS, root_vals[0] / NUM_BUCKETS),
                    jnp.full(NUM_BUCKETS, root_vals[1] / NUM_BUCKETS),
                ]
            )
            buf_new, rng_after_append = append_one(buf, pbs_enc, values_target, rng_act)
            action, _ = _get_action_from_model(
                env, env_st, value_fn, config["max_depth"], rng_after_append
            )
            next_state, _, rewards, _, done, _ = env.step(rng_step, env_st, action)
            use_step = has_children & (~hand_done)
            new_env_st = jax.tree.map(
                lambda n, e: jnp.where(use_step, n, e),
                next_state,
                env_st,
            )
            new_buf = jax.tree.map(
                lambda n, e: jnp.where(use_step, n, e),
                buf_new,
                buf,
            )
            new_rng = lax.cond(use_step, lambda: rng_new, lambda: rng_key)
            new_hand_done = hand_done | (use_step & done)
            return (new_env_st, new_buf, ts, new_rng, new_hand_done), rewards

        def hand_cond(carry):
            env_st, _, _, _, hand_done = carry
            has_children = jnp.any(env.get_avail_actions(env_st))
            return (~hand_done) & has_children

        def hand_body(carry):
            new_carry, _ = hand_step(carry, None)
            return new_carry

        def value_train_epoch(carry, _):
            ts, buf, rng_key = carry
            rng_key, rng_b = jax.random.split(rng_key)
            pbs_b, values_b = sample_batch(rng_b, buf, config["batch_size"])

            def loss_fn(p):
                pred = network.apply(p, pbs_b)
                return jnp.mean(huber_loss(pred - values_b))

            loss, grads = jax.value_and_grad(loss_fn)(ts.params)
            ts = ts.apply_gradients(grads=grads)
            return (ts, buf, rng_key), loss

        def compare_eval(carry, h):
            rng_key, train_params, baseline_params = carry
            rng_key, rng_reset = jax.random.split(rng_key)
            env_st, _ = env.reset(rng_reset)
            swap = (h % 2) == 1
            train_player_idx = jnp.where(swap, 1, 0)

            def play_cond(c):
                _, _, _, step_count, episode_done = c
                return (~episode_done) & (step_count < 200)

            def play_body(c):
                env_st_inner, rng_inner, terminal_rewards, step_count, episode_done = c
                cur = env_st_inner.current_player_idx
                is_train_turn = cur == train_player_idx

                def train_turn_step(rng_action: PRNGKeyArray):
                    return _get_action_from_params(
                        env,
                        env_st_inner,
                        network,
                        train_params,
                        config["max_depth"],
                        rng_action,
                    )

                def baseline_turn_step(rng_action: PRNGKeyArray):
                    if compare_against_random:
                        avail = env.get_avail_actions(env_st_inner)
                        logits = jnp.where(avail, 0.0, -1e9)
                        rng_next, rng_choice = jax.random.split(rng_action)
                        action = jax.random.categorical(rng_choice, logits)
                        return action, rng_next
                    return _get_action_from_params(
                        env,
                        env_st_inner,
                        network,
                        baseline_params,
                        config["max_depth"],
                        rng_action,
                    )

                action, rng_inner = lax.cond(
                    is_train_turn,
                    train_turn_step,
                    baseline_turn_step,
                    rng_inner,
                )
                rng_inner, rng_s = jax.random.split(rng_inner)
                next_st, _, rewards, absorbing, done, _ = env.step(
                    rng_s, env_st_inner, action
                )
                next_st = next_st.replace(absorbing=absorbing)
                # Evaluation metric should use terminal episode return only.
                new_terminal_rewards = jnp.where(done, rewards, terminal_rewards)
                return (
                    next_st,
                    rng_inner,
                    new_terminal_rewards,
                    step_count + 1,
                    episode_done | done,
                )

            (_, rng_key, terminal_rewards, episode_len, episode_done) = lax.while_loop(
                play_cond,
                play_body,
                (
                    env_st,
                    rng_key,
                    jnp.zeros(2, dtype=jnp.float32),
                    jnp.array(0, dtype=jnp.int32),
                    jnp.bool_(False),
                ),
            )
            train_chips = lax.cond(
                swap,
                lambda: terminal_rewards[1],
                lambda: terminal_rewards[0],
            )
            return (
                (rng_key, train_params, baseline_params),
                (
                    train_chips,
                    episode_len.astype(jnp.float32),
                    episode_done.astype(jnp.float32),
                ),
            )

        def update_step(carry, _):
            runner_state = carry
            env_st = runner_state.env_state
            buf = runner_state.replay_buffer
            ts = runner_state.value_train_state
            rng_key = runner_state.rng

            train_value_fn = lambda enc: network.apply(ts.params, enc)
            (env_st, buf, ts, rng_key, _) = lax.while_loop(
                hand_cond,
                hand_body,
                (env_st, buf, ts, rng_key, jnp.bool_(False)),
            )
            rng_key, rng_reset = jax.random.split(rng_key)
            env_st, _ = env.reset(rng_reset)

            rng_key, rng_train = jax.random.split(rng_key)
            can_train = buf.size >= config["batch_size"]

            def do_value_train():
                (ts_new, _, _), losses = lax.scan(
                    value_train_epoch,
                    (ts, buf, rng_train),
                    jnp.arange(config["num_value_train_steps"]),
                )
                return ts_new, jnp.mean(losses)

            def no_value_train():
                return ts, jnp.array(0.0, dtype=jnp.float32)

            ts, mean_loss = lax.cond(can_train, do_value_train, no_value_train)

            should_compare = (
                runner_state.update_step % config["compare_interval"] == 0
            ) & config["compare"]

            def do_compare():
                (_, _, _), (chips_arr, episode_len_arr, done_arr) = lax.scan(
                    compare_eval,
                    (rng_key, ts.params, baseline_params),
                    jnp.arange(config["compare_steps"]),
                )
                done_count = jnp.maximum(jnp.sum(done_arr), 1.0)
                avg_terminal_rewards = jnp.sum(chips_arr) / done_count
                avg_episode_len = jnp.sum(episode_len_arr) / done_count
                return avg_terminal_rewards, avg_episode_len

            avg_chips, avg_eval_episode_length = lax.cond(
                should_compare,
                do_compare,
                lambda: (
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                ),
            )

            def log_callback(sid, m, st):
                np_m = {k: np.array(v) for k, v in m.items()}
                if int(st) % config["log_interval"] == 0 and LOGGER is not None:
                    LOGGER.log(int(sid), np_m)

            metric = {
                "value_loss": mean_loss,
                "replay_size": buf.size.astype(jnp.float32),
                "replay_seen": buf.seen.astype(jnp.float32),
                "update_step": runner_state.update_step.astype(jnp.float32),
                "avg_chips_vs_baseline": (
                    jnp.array(0.0, dtype=jnp.float32)
                    if compare_against_random
                    else avg_chips
                ),
                "avg_chips_vs_random": (
                    avg_chips
                    if compare_against_random
                    else jnp.array(0.0, dtype=jnp.float32)
                ),
                "avg_eval_episode_length": avg_eval_episode_length,
            }
            jax.experimental.io_callback(
                log_callback,
                None,
                exp_id,
                metric,
                runner_state.update_step,
            )

            return RunnerState(
                value_train_state=ts,
                replay_buffer=buf,
                env_state=env_st,
                rng=rng_key,
                update_step=runner_state.update_step + 1,
            )

        initial_state = RunnerState(
            value_train_state=train_state,
            replay_buffer=replay_buffer,
            env_state=env_state,
            rng=rng,
            update_step=jnp.array(0, dtype=jnp.int32),
        )

        final_state = lax.scan(
            lambda c, _: (update_step(c, None), ()),
            initial_state,
            jnp.arange(config["num_update_steps"]),
        )[0]

        return {"params": final_state.value_train_state.params}

    return train


@hydra.main(version_base=None, config_path="./", config_name="config_rebel")
def main(config: dict) -> None:
    try:
        config = OmegaConf.to_container(config)

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
        result = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
        if config["save_final"] and "params" in result:
            os.makedirs(config["save_dir"], exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                config["save_dir"], f"hul_rebel_{timestamp}.msgpack"
            )
            params = result["params"]
            if config["num_seeds"] > 1:
                params = jax.tree.map(lambda x: x[0], params)
            bytes_output = serialization.to_bytes(params)
            with open(save_path, "wb") as f:
                f.write(bytes_output)
            print(f"Saved model to {save_path}")
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
