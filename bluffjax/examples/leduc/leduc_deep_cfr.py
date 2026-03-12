"""
Deep CFR for Leduc Poker (OpenSpiel-style).

Uses the Leduc implementation and exploitability from new_test.py.
Based on https://arxiv.org/abs/1811.00164.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bluffjax.utils.game_utils import leduc_exploitability as leduc

# --- Constants ---
ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min
NUM_ACTIONS = 3  # fold, call, raise
EMBEDDING_SIZE = 36  # Leduc obs dim from new_test


def _legal_actions_mask(state: leduc.LeducState) -> np.ndarray:
    """Boolean mask of shape (3,) for legal actions."""
    legal = state.legal_actions()
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for a in legal:
        mask[a] = True
    return mask


def _information_state_tensor(state: leduc.LeducState) -> np.ndarray:
    """36-dim observation for the current player (at a decision node)."""
    return leduc._state_to_obs(state).astype(np.float32)


# --- Reservoir buffer (Python-based for portability) ---
class AdvantageMemory(NamedTuple):
    info_state: np.ndarray
    iteration: int
    advantage: np.ndarray
    legal_mask: np.ndarray


class StrategyMemory(NamedTuple):
    info_state: np.ndarray
    iteration: int
    strategy_action_probs: np.ndarray
    legal_mask: np.ndarray


class ReservoirBuffer:
    """Reservoir sampling buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list = []
        self.add_calls = 0

    def append(self, item: NamedTuple) -> None:
        self.add_calls += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = random.randint(0, self.add_calls - 1)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, n: int) -> tuple:
        if not self.buffer:
            return None
        size = min(n, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), size)
        if isinstance(self.buffer[0], AdvantageMemory):
            return AdvantageMemory(
                info_state=np.stack([self.buffer[i].info_state for i in indices]),
                iteration=np.stack(
                    [np.array(self.buffer[i].iteration) for i in indices]
                ),
                advantage=np.stack([self.buffer[i].advantage for i in indices]),
                legal_mask=np.stack([self.buffer[i].legal_mask for i in indices]),
            )
        else:
            return StrategyMemory(
                info_state=np.stack([self.buffer[i].info_state for i in indices]),
                iteration=np.stack(
                    [np.array(self.buffer[i].iteration) for i in indices]
                ),
                strategy_action_probs=np.stack(
                    [self.buffer[i].strategy_action_probs for i in indices]
                ),
                legal_mask=np.stack([self.buffer[i].legal_mask for i in indices]),
            )

    def __len__(self) -> int:
        return len(self.buffer)


# --- MLP (flax.linen) ---
class AdvantageMLP(nn.Module):
    """MLP that outputs raw advantage logits (no softmax)."""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class PolicyMLP(nn.Module):
    """MLP that outputs raw logits (softmax applied externally with masking)."""

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


# --- Deep CFR Solver ---
class DeepCFRSolver:
    def __init__(
        self,
        policy_hidden: int = 64,
        advantage_hidden: int = 64,
        num_iterations: int = 100,
        num_traversals: int = 20,
        learning_rate: float = 1e-3,
        batch_size_advantage: int = 256,
        batch_size_strategy: int = 256,
        memory_capacity: int = 100_000,
        policy_network_train_steps: int = 500,
        advantage_network_train_steps: int = 200,
        reinitialize_advantage_networks: bool = False,
        seed: int = 42,
    ) -> None:
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._num_players = 2
        self._num_actions = NUM_ACTIONS
        self._embedding_size = EMBEDDING_SIZE
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._iteration = 1
        self._learning_rate = learning_rate
        self._numpy_rng = np.random.default_rng(seed)
        self._memory_capacity = int(memory_capacity)

        # Initialize networks
        rng_key = jax.random.PRNGKey(seed)
        dummy_obs = jnp.zeros((1, EMBEDDING_SIZE), dtype=jnp.float32)

        self._advantage_networks = [
            AdvantageMLP(hidden_dim=advantage_hidden) for _ in range(self._num_players)
        ]
        self._advantage_params = [
            self._advantage_networks[p].init(rng_key, dummy_obs)
            for p in range(self._num_players)
        ]
        self._advantage_opt = [
            optax.adam(learning_rate) for _ in range(self._num_players)
        ]
        self._advantage_opt_state = [
            self._advantage_opt[p].init(self._advantage_params[p])
            for p in range(self._num_players)
        ]

        self._policy_network = PolicyMLP(hidden_dim=policy_hidden)
        self._policy_params = self._policy_network.init(rng_key, dummy_obs)
        self._policy_opt = optax.adam(learning_rate)
        self._policy_opt_state = self._policy_opt.init(self._policy_params)

        self._advantage_memories: list[ReservoirBuffer | None] = [None, None]
        self._strategy_memories: ReservoirBuffer | None = None
        self._cached_policy: dict = {}

        # JIT-compiled forward passes (same architecture for both players)
        adv_net = self._advantage_networks[0]
        self._advantage_apply_jit = jax.jit(
            lambda params, obs: adv_net.apply(params, obs[jnp.newaxis, :])[0]
        )
        self._policy_apply_jit = jax.jit(
            lambda params, obs: self._policy_network.apply(params, obs[jnp.newaxis, :])[
                0
            ]
        )

        # JIT-compiled training steps
        def _advantage_train_step(
            params, opt_state, info_state, advantage, legal_mask, iteration
        ):
            def loss_fn(p):
                logits = jax.vmap(lambda x: adv_net.apply(p, x[jnp.newaxis, :])[0])(
                    info_state
                )
                logits = jnp.where(legal_mask, logits, 0.0)
                it = jnp.sqrt(jnp.asarray(iteration, dtype=jnp.float32))[:, jnp.newaxis]
                return jnp.mean(optax.l2_loss(logits * it, advantage * it))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self._advantage_opt[0].update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        def _policy_train_step(
            params, opt_state, info_state, strategy_probs, legal_mask, iteration
        ):
            def loss_fn(p):
                logits = jax.vmap(
                    lambda x: self._policy_network.apply(p, x[jnp.newaxis, :])[0]
                )(info_state)
                logits = jnp.where(legal_mask, logits, ILLEGAL_ACTION_LOGITS_PENALTY)
                probs = jax.nn.softmax(logits)
                it = jnp.sqrt(jnp.asarray(iteration, dtype=jnp.float32))[:, jnp.newaxis]
                return jnp.mean(optax.l2_loss(probs * it, strategy_probs * it))

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = self._policy_opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        self._advantage_train_step_jit = jax.jit(_advantage_train_step)
        self._policy_train_step_jit = jax.jit(_policy_train_step)

    def _sample_action_from_advantage(
        self, state: leduc.LeducState, player: int
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = jnp.asarray(_information_state_tensor(state), dtype=jnp.float32)
        logits = np.asarray(
            self._advantage_apply_jit(self._advantage_params[player], obs)
        )
        legal_mask = _legal_actions_mask(state)
        logits = np.where(legal_mask, np.asarray(logits), -np.inf)
        advantages = np.maximum(logits, 0.0)
        if advantages.sum() <= 0:
            legal = state.legal_actions()
            matched = np.zeros(NUM_ACTIONS)
            matched[legal[0]] = 1.0
        else:
            matched = advantages / advantages.sum()
        return advantages, matched

    def _traverse_game_tree(self, state: leduc.LeducState, player: int) -> np.floating:
        if state.is_terminal():
            return np.array(state.returns()[player], dtype=np.float32)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = self._numpy_rng.choice(np.asarray(outcomes), p=np.asarray(probs))
            return self._traverse_game_tree(state.child(int(action)), player)

        cur = state.current_player
        if cur == player:
            _, strategy = self._sample_action_from_advantage(state, player)
            strategy = np.array(strategy, dtype=np.float64)
            exp_payoff = np.zeros(self._num_actions, dtype=np.float64)
            for action in state.legal_actions():
                exp_payoff[action] = self._traverse_game_tree(
                    state.child(action), player
                )
            cfv = np.sum(exp_payoff * strategy)
            samp_regret = (exp_payoff - cfv) * _legal_actions_mask(state)
            data = AdvantageMemory(
                info_state=_information_state_tensor(state),
                iteration=self._iteration,
                advantage=samp_regret.astype(np.float32),
                legal_mask=_legal_actions_mask(state),
            )
            if self._advantage_memories[player] is None:
                self._advantage_memories[player] = ReservoirBuffer(
                    self._memory_capacity
                )
            self._advantage_memories[player].append(data)
            return np.float64(cfv)
        else:
            other = cur
            _, strategy = self._sample_action_from_advantage(state, other)
            probs = np.array(strategy, dtype=np.float64)
            probs /= probs.sum()
            legal = state.legal_actions()
            probs_legal = np.array([probs[a] for a in legal], dtype=np.float64)
            probs_legal /= probs_legal.sum()
            sampled_action = self._numpy_rng.choice(np.asarray(legal), p=probs_legal)
            data = StrategyMemory(
                info_state=_information_state_tensor(state),
                iteration=self._iteration,
                strategy_action_probs=probs.astype(np.float32),
                legal_mask=_legal_actions_mask(state),
            )
            if self._strategy_memories is None:
                self._strategy_memories = ReservoirBuffer(self._memory_capacity)
            self._strategy_memories.append(data)
            return self._traverse_game_tree(state.child(int(sampled_action)), player)

    def policy_for_exploitability(
        self,
    ) -> Callable[[leduc.LeducState], dict[int, float]]:
        """Build policy(state) -> dict[int, float] for exploitability."""

        def policy(state: leduc.LeducState) -> dict[int, float]:
            legal = state.legal_actions()
            if not legal:
                return {}
            obs = _information_state_tensor(state)
            key = obs.tobytes()
            if key in self._cached_policy:
                logits = self._cached_policy[key]
            else:
                obs_jax = jnp.asarray(obs, dtype=jnp.float32)
                logits = np.asarray(
                    self._policy_apply_jit(self._policy_params, obs_jax)
                )
                self._cached_policy[key] = logits
            mask = np.ones(NUM_ACTIONS, dtype=np.float64) * (-np.inf)
            for a in legal:
                mask[a] = float(logits[a])
            expo = np.exp(mask - np.max(mask))
            probs = expo / expo.sum()
            return {a: float(probs[a]) for a in legal}

        return policy

    def _learn_advantage_network(self, player: int) -> float | None:
        buf = self._advantage_memories[player]
        if buf is None or len(buf) == 0:
            return None
        batch_size = min(self._batch_size_advantage, len(buf))
        batch = buf.sample(batch_size)
        info_state = jnp.asarray(batch.info_state, dtype=jnp.float32)
        advantage = jnp.asarray(batch.advantage, dtype=jnp.float32)
        legal_mask = jnp.asarray(batch.legal_mask, dtype=jnp.bool_)
        iteration = jnp.asarray(batch.iteration, dtype=jnp.int32)
        (
            self._advantage_params[player],
            self._advantage_opt_state[player],
            loss,
        ) = self._advantage_train_step_jit(
            self._advantage_params[player],
            self._advantage_opt_state[player],
            info_state,
            advantage,
            legal_mask,
            iteration,
        )
        return float(loss)

    def _learn_strategy_network(self) -> float | None:
        if self._strategy_memories is None or len(self._strategy_memories) == 0:
            return None
        batch_size = min(self._batch_size_strategy, len(self._strategy_memories))
        batch = self._strategy_memories.sample(batch_size)
        info_state = jnp.asarray(batch.info_state, dtype=jnp.float32)
        strategy_probs = jnp.asarray(batch.strategy_action_probs, dtype=jnp.float32)
        legal_mask = jnp.asarray(batch.legal_mask, dtype=jnp.bool_)
        iteration = jnp.asarray(batch.iteration, dtype=jnp.int32)
        (
            self._policy_params,
            self._policy_opt_state,
            loss,
        ) = self._policy_train_step_jit(
            self._policy_params,
            self._policy_opt_state,
            info_state,
            strategy_probs,
            legal_mask,
            iteration,
        )
        return float(loss)


def _exploitability(policy_fn: Callable[[leduc.LeducState], dict[int, float]]) -> float:
    """Exploitability using new_test's BestResponseSolver and _state_values."""
    root = leduc.initial_state()
    on_policy = leduc._state_values(root, policy_fn)
    br0 = leduc.BestResponseSolver(root, 0, policy_fn).value(root)
    br1 = leduc.BestResponseSolver(root, 1, policy_fn).value(root)
    nash_conv = (br0 - on_policy[0]) + (br1 - on_policy[1])
    return nash_conv / 2.0


def run_deep_cfr(
    num_iterations: int = 50,
    num_traversals: int = 100,
    log_interval: int = 5,
    **kwargs,
) -> list[tuple[int, float]]:
    solver = DeepCFRSolver(
        policy_hidden=64,
        advantage_hidden=64,
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        batch_size_advantage=min(256, 936 * num_traversals // 10),
        batch_size_strategy=min(256, 936 * num_traversals // 10),
        memory_capacity=100_000,
        policy_network_train_steps=500,
        advantage_network_train_steps=200,
        reinitialize_advantage_networks=True,
        seed=42,
        **kwargs,
    )

    root = leduc.initial_state()
    exploitability_log: list[tuple[int, float]] = []

    for it in range(num_iterations):
        if (it + 1) % log_interval == 0 or it == 0:
            solver._learn_strategy_network()
            pol = solver.policy_for_exploitability()
            expl = _exploitability(pol)
            exploitability_log.append((solver._iteration, float(expl)))
            print(f"Iteration {solver._iteration:4d}: exploitability = {expl:.6f}")
            solver._cached_policy.clear()

        for p in range(2):
            for _ in range(num_traversals):
                solver._traverse_game_tree(root, p)
            if solver._reinitialize_advantage_networks:
                rng_key = jax.random.PRNGKey(solver._iteration * 2 + p)
                dummy = jnp.zeros((1, EMBEDDING_SIZE), dtype=jnp.float32)
                solver._advantage_params[p] = solver._advantage_networks[p].init(
                    rng_key, dummy
                )
                solver._advantage_opt_state[p] = solver._advantage_opt[p].init(
                    solver._advantage_params[p]
                )
            solver._learn_advantage_network(p)

        solver._iteration += 1

    solver._learn_strategy_network()
    pol = solver.policy_for_exploitability()
    final_expl = _exploitability(pol)
    exploitability_log.append((solver._iteration, float(final_expl)))
    print(
        f"Iteration {solver._iteration:4d}: exploitability = {final_expl:.6f} (final)"
    )

    return exploitability_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep CFR on Leduc Poker (OpenSpiel-style)"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--traversals", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=5)
    args = parser.parse_args()

    print("Deep CFR for Leduc Poker (OpenSpiel-style)")
    print("=" * 50)
    expl_log = run_deep_cfr(
        num_iterations=args.iterations,
        num_traversals=args.traversals,
        log_interval=args.log_interval,
    )
    print("=" * 50)
    print(f"Final exploitability: {expl_log[-1][1]:.6f}")


if __name__ == "__main__":
    main()
