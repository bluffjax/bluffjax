"""
Deep CFR for Kuhn Poker.

Standalone implementation based on https://arxiv.org/abs/1811.00164.
Uses neural networks (advantage + strategy) with reservoir buffers.
"""

import collections
import functools
from typing import Callable, Iterable, NamedTuple

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bluffjax.utils.game_utils.kuhn_exploitability import (
    exploitability,
    get_current_player,
    get_legal_actions,
    infoset_key,
    is_terminal,
    get_returns,
)

# --- Constants ---
ILLEGAL_ACTION_LOGITS_PENALTY = jnp.finfo(jnp.float32).min
NUM_ACTIONS = 2
EMBEDDING_SIZE = 9  # Kuhn obs: 3 (card) + 2 + 2 + 2


# --- Info state encoding: OpenSpiel infoset key -> 9-dim vector ---
def _infoset_to_obs(key: str) -> np.ndarray:
    """Convert infoset key to 9-dim observation (matches KuhnPoker.obs_from_state)."""
    card = int(key[0])
    betstr = key[1:]

    agent_hand = np.zeros(3, dtype=np.float32)
    agent_hand[card] = 1.0

    if not betstr:  # P0 first to act
        self_chips, opp_chips = 1, 1
        player_rel = 0
    elif betstr == "pb":  # P0 facing bet
        self_chips, opp_chips = 1, 2
        player_rel = 0
    elif betstr == "p":  # P1 after P0 pass
        self_chips, opp_chips = 1, 1
        player_rel = 1
    else:  # betstr == "b", P1 after P0 bet
        self_chips, opp_chips = 1, 2
        player_rel = 1

    self_oh = np.zeros(2, dtype=np.float32)
    self_oh[self_chips - 1] = 1.0
    opp_oh = np.zeros(2, dtype=np.float32)
    opp_oh[opp_chips - 1] = 1.0
    player_oh = np.zeros(2, dtype=np.float32)
    player_oh[player_rel] = 1.0

    return np.concatenate([agent_hand, self_oh, opp_oh, player_oh])


# --- Kuhn game state for traversal ---
class KuhnState:
    """Minimal game state for Deep CFR traversal. hands=None means chance node."""

    __slots__ = ("hands", "history")

    def __init__(self, hands: tuple[int, int] | None, history: tuple[int, ...]):
        self.hands = hands
        self.history = history

    def is_terminal(self) -> bool:
        return self.hands is not None and is_terminal(self.history)

    def is_chance_node(self) -> bool:
        return self.hands is None

    def current_player(self) -> int:
        return get_current_player(self.history)

    def returns(self) -> tuple[float, float]:
        return get_returns(self.hands, self.history)

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Deal: 6 outcomes (0,1),(0,2),(1,0),(1,2),(2,0),(2,1), each prob 1/6."""
        import itertools

        deals = list(itertools.permutations([0, 1, 2], 2))
        return [(i, 1.0 / 6.0) for i in range(6)]

    def _deal_from_index(self, idx: int) -> tuple[int, int]:
        import itertools

        deals = list(itertools.permutations([0, 1, 2], 2))
        return deals[idx]

    def child(self, action: int) -> "KuhnState":
        if self.hands is None:
            hands = self._deal_from_index(action)
            return KuhnState(hands, ())
        return KuhnState(self.hands, self.history + (action,))

    def legal_actions(self) -> list[int]:
        return get_legal_actions(self.history)

    def legal_actions_mask(self, player: int) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for a in get_legal_actions(self.history):
            mask[a] = True
        return mask

    def information_state_tensor(self, player: int) -> np.ndarray:
        card = self.hands[player]
        infok = infoset_key(card, self.history)
        return _infoset_to_obs(infok).astype(np.float32)


def _root_state() -> KuhnState:
    return KuhnState(None, ())


# --- Reservoir buffer (from OpenSpiel deep_cfr) ---
class AdvantageMemory(NamedTuple):
    info_state: chex.Array
    iteration: chex.Array
    advantage: chex.Array
    legal_mask: chex.Array


class StrategyMemory(NamedTuple):
    info_state: chex.Array
    iteration: chex.Array
    strategy_action_probs: chex.Array
    legal_mask: chex.Array


class ReservoirBufferState(NamedTuple):
    experience: chex.ArrayTree
    capacity: chex.Numeric
    add_calls: chex.Array
    is_full: chex.Array

    def __len__(self) -> int:
        return int(jnp.where(self.is_full, self.capacity, self.add_calls))


class ReservoirBuffer:
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("capacity",))
    def init(capacity: chex.Numeric, experience: chex.ArrayTree) -> ReservoirBufferState:
        experience = jax.tree.map(jnp.empty_like, experience)
        experience = jax.tree.map(
            lambda x: jnp.broadcast_to(x[jnp.newaxis, ...], (capacity, *x.shape)),
            experience,
        )
        return ReservoirBufferState(
            capacity=capacity,
            experience=experience,
            add_calls=jnp.array(0),
            is_full=jnp.array(False, dtype=jnp.bool),
        )

    @staticmethod
    @functools.partial(jax.jit, donate_argnums=(0,))
    def append(
        state: ReservoirBufferState, experience: chex.ArrayTree, rng: chex.PRNGKey
    ) -> ReservoirBufferState:
        idx = jax.random.randint(rng, (), 0, state.add_calls + 1)
        is_full = state.is_full | (state.add_calls >= state.capacity)
        write_idx = jnp.where(is_full, idx, state.add_calls)
        should_update = write_idx < state.capacity

        def update_leaf(buffer_leaf, exp_leaf):
            new_val = jnp.where(should_update, exp_leaf, buffer_leaf[write_idx])
            return buffer_leaf.at[write_idx].set(new_val)

        new_experience = jax.tree.map(update_leaf, state.experience, experience)
        return ReservoirBufferState(
            capacity=state.capacity,
            experience=new_experience,
            add_calls=state.add_calls + 1,
            is_full=is_full,
        )

    @staticmethod
    @functools.partial(jax.jit, static_argnames="num_samples")
    def sample(rng: chex.PRNGKey, state: ReservoirBufferState, num_samples: int) -> chex.ArrayTree:
        max_size = jnp.where(state.is_full, state.capacity, state.add_calls)
        indices = jax.random.randint(rng, shape=(num_samples,), minval=0, maxval=max_size)
        return jax.tree.map(lambda x: x[indices], state.experience)


# --- LRU cache (from OpenSpiel) ---
class LRUCache:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._data: collections.OrderedDict = collections.OrderedDict()

    def clear(self) -> None:
        self._data.clear()

    def make(self, key, fn: Callable):
        try:
            val = self._data.pop(key)
        except KeyError:
            val = fn()
            if len(self._data) >= self._max_size:
                self._data.popitem(last=False)
        self._data[key] = val
        return val


# --- MLP (from OpenSpiel deep_cfr) ---
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int,
        final_activation: Callable = lambda x: x,
        seed: int = 0,
    ) -> None:
        layers_ = []

        def _create_linear_block(in_features, out_features, act=nn.relu):
            return nn.Sequential(
                nn.Linear(
                    in_features,
                    out_features,
                    kernel_init=nn.initializers.glorot_uniform(),
                    rngs=nn.Rngs(seed),
                ),
                act,
            )

        for size in hidden_sizes:
            layers_.append(_create_linear_block(input_size, size, act=nn.relu))
            input_size = size
        layers_.append(nn.LayerNorm(input_size, rngs=nn.Rngs(seed)))
        layers_.append(_create_linear_block(input_size, output_size, act=lambda x: x))
        if final_activation:
            layers_.append(final_activation)
        self.model = nn.Sequential(*layers_)

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.model(x)


@nn.vmap(in_axes=(None, 0), out_axes=0)
def _forward(model, x):
    return model(x)


# --- Deep CFR Solver ---
class DeepCFRSolver:
    def __init__(
        self,
        policy_network_layers: tuple[int, ...] = (64, 64),
        advantage_network_layers: tuple[int, ...] = (64, 64),
        num_iterations: int = 100,
        num_traversals: int = 20,
        learning_rate: float = 1e-3,
        batch_size_advantage: int = 256,
        batch_size_strategy: int = 256,
        memory_capacity: int = 100_000,
        policy_network_train_steps: int = 1000,
        advantage_network_train_steps: int = 200,
        reinitialize_advantage_networks: bool = True,
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
        self._rngkey = jax.random.key(seed)
        self._memory_capacity = int(memory_capacity)

        self._advantage_memories: list[ReservoirBufferState | None] = [None, None]
        self._advantage_networks = [
            MLP(
                self._embedding_size,
                list(advantage_network_layers),
                self._num_actions,
                lambda x: x,
                seed + p,
            )
            for p in range(self._num_players)
        ]
        self._advantage_opt = [
            nn.Optimizer(
                self._advantage_networks[p],
                optax.adam(self._learning_rate),
                wrt=nn.Param,
            )
            for p in range(self._num_players)
        ]
        self._empty_advantage_states = [
            nn.state((self._advantage_networks[p], self._advantage_opt[p]))
            for p in range(self._num_players)
        ]
        self._advantage_graphdefs = [
            nn.graphdef((self._advantage_networks[p], self._advantage_opt[p]))
            for p in range(self._num_players)
        ]

        self._strategy_memories: ReservoirBufferState | None = None
        self._policy_network = MLP(
            self._embedding_size,
            list(policy_network_layers),
            self._num_actions,
            lambda x: x,
            seed,
        )
        self._policy_opt = nn.Optimizer(
            self._policy_network, optax.adam(self._learning_rate), wrt=nn.Param
        )
        self._empty_policy_state = nn.state((self._policy_network, self._policy_opt))
        self._policy_graphdef = nn.graphdef((self._policy_network, self._policy_opt))

        self._advantage_loss = self._policy_loss = jax.vmap(optax.l2_loss)
        self._jittable_matched_regrets = self._get_jittable_matched_regrets()
        self._jittable_adv_update = self._make_train_step(self._get_jittable_adv_update())
        self._jittable_policy_update = self._make_train_step(self._get_jittable_policy_update())
        self._cached_policy = LRUCache(2**16)

    def _get_jittable_adv_update(self) -> Callable:
        def _loss_adv(
            advantage_model: nn.Module,
            info_states: chex.Array,
            samp_regrets: chex.Array,
            masks: chex.Array,
            iterations: chex.Array,
        ) -> chex.Array:
            preds = _forward(advantage_model, info_states)
            preds = jnp.where(masks, preds, jnp.array(0))
            it = jnp.sqrt(iterations)
            return self._advantage_loss(preds * it, samp_regrets * it).mean()

        def update(
            advantage_model: nn.Module,
            optimiser: nn.Optimizer,
            batch: AdvantageMemory,
        ) -> chex.Array:
            main_loss, grads = nn.value_and_grad(_loss_adv)(
                advantage_model,
                batch.info_state,
                batch.advantage,
                batch.legal_mask,
                batch.iteration,
            )
            optimiser.update(advantage_model, grads)
            return main_loss

        return update

    def _get_jittable_policy_update(self) -> Callable:
        def _loss_policy(
            policy_model: nn.Module,
            info_states: chex.Array,
            action_probs: chex.Array,
            masks: chex.Array,
            iterations: chex.Array,
        ) -> chex.Array:
            preds = _forward(policy_model, info_states)
            preds = jnp.where(masks, preds, ILLEGAL_ACTION_LOGITS_PENALTY)
            preds = nn.softmax(preds)
            it = jnp.sqrt(iterations)
            return self._policy_loss(preds * it, action_probs * it).mean()

        def update(
            policy_model: nn.Module,
            optimiser: nn.Optimizer,
            batch: StrategyMemory,
        ) -> chex.Array:
            main_loss, grads = nn.value_and_grad(_loss_policy)(
                policy_model,
                batch.info_state,
                batch.strategy_action_probs,
                batch.legal_mask,
                batch.iteration,
            )
            optimiser.update(policy_model, grads)
            return main_loss

        return update

    def _get_jittable_matched_regrets(self) -> Callable:
        @functools.partial(jax.jit, static_argnames=("graphdef",))
        def get_matched_regrets(
            graphdef: nn.GraphDef,
            state: nn.State,
            info_state: chex.Array,
            legal_actions_mask: chex.Array,
        ) -> tuple[chex.Array, chex.Array]:
            advantage_model = nn.merge(graphdef, state)
            advs = advantage_model(info_state)
            advs = jnp.where(legal_actions_mask, advs, ILLEGAL_ACTION_LOGITS_PENALTY)
            advantages = nn.relu(advs)
            summed_regret = jnp.sum(advantages)
            matched_regrets = jnp.where(
                summed_regret > 0,
                advantages / summed_regret,
                jax.nn.one_hot(jnp.argmax(advs), self._num_actions),
            )
            return advantages, matched_regrets

        return get_matched_regrets

    def _make_train_step(self, update_fn: Callable) -> Callable:
        @jax.jit
        def _train_step(graphdef, state, batch) -> tuple:
            model, optimiser = nn.merge(graphdef, state, copy=True)
            loss = update_fn(model, optimiser, batch)
            state = nn.state((model, optimiser))
            return (state, loss)

        return _train_step

    def _next_rng_key(self) -> chex.PRNGKey:
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey

    def _reinitialize_policy_network(self) -> None:
        nn.update((self._policy_network, self._policy_opt), self._empty_policy_state)

    def _reinitialize_advantage_network(self, player: int) -> None:
        nn.update(
            (self._advantage_networks[player], self._advantage_opt[player]),
            self._empty_advantage_states[player],
        )

    def _append_to_advantage_buffer(self, player: int, data: AdvantageMemory) -> None:
        if self._advantage_memories[player] is None:
            self._advantage_memories[player] = ReservoirBuffer.init(self._memory_capacity, data)
        self._advantage_memories[player] = ReservoirBuffer.append(
            self._advantage_memories[player], data, self._next_rng_key()
        )

    def _append_to_strategy_buffer(self, data: StrategyMemory) -> None:
        if self._strategy_memories is None:
            self._strategy_memories = ReservoirBuffer.init(self._memory_capacity, data)
        self._strategy_memories = ReservoirBuffer.append(
            self._strategy_memories, data, self._next_rng_key()
        )

    def _traverse_game_tree(self, state: KuhnState, player: int) -> np.ndarray:
        if state.is_terminal():
            return np.array(state.returns()[player], dtype=np.float32)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(np.asarray(outcomes), p=np.asarray(probs))
            return self._traverse_game_tree(state.child(action), player)

        cur = state.current_player()
        if cur == player:
            _, strategy = self._sample_action_from_advantage(state, player)
            strategy = np.array(strategy)
            exp_payoff = np.zeros(self._num_actions, dtype=np.float64)
            for action in state.legal_actions():
                exp_payoff[action] = self._traverse_game_tree(state.child(action), player)
            cfv = np.sum(exp_payoff * strategy)
            samp_regret = (exp_payoff - cfv) * state.legal_actions_mask(player)
            data = AdvantageMemory(
                jnp.asarray(state.information_state_tensor(player), dtype=jnp.float32),
                jnp.asarray(self._iteration, dtype=jnp.int32).reshape(1),
                jnp.asarray(samp_regret, dtype=jnp.float32),
                jnp.asarray(state.legal_actions_mask(player), dtype=jnp.bool),
            )
            self._append_to_advantage_buffer(player, data)
            return np.float64(cfv)
        else:
            other = cur
            _, strategy = self._sample_action_from_advantage(state, other)
            probs = np.array(strategy, dtype=np.float64)
            probs /= probs.sum()
            sampled_action = np.random.choice(np.arange(self._num_actions), p=probs)
            data = StrategyMemory(
                jnp.asarray(state.information_state_tensor(other), dtype=jnp.float32),
                jnp.asarray(self._iteration, dtype=jnp.int32).reshape(-1),
                jnp.asarray(probs, dtype=jnp.float32),
                jnp.asarray(state.legal_actions_mask(other), dtype=jnp.bool),
            )
            self._append_to_strategy_buffer(data)
            return self._traverse_game_tree(state.child(int(sampled_action)), player)

    def _sample_action_from_advantage(
        self, state: KuhnState, player: int
    ) -> tuple[chex.Array, chex.Array]:
        self._advantage_networks[player].eval()
        info_state = jnp.asarray(state.information_state_tensor(player), dtype=jnp.float32)
        legal_mask = jnp.asarray(state.legal_actions_mask(player), dtype=jnp.bool)
        graphdef, nn_state = nn.split(self._advantage_networks[player])
        advantages, matched_regrets = self._jittable_matched_regrets(
            graphdef, nn_state, info_state, legal_mask
        )
        return advantages, matched_regrets

    def action_probabilities(self, state: KuhnState) -> dict[int, chex.Array]:
        """Policy network output for current state. Returns {action: prob}."""
        self._policy_network.eval()
        cur = state.current_player()
        legal = state.legal_actions()
        info_vec = np.asarray(state.information_state_tensor(cur), dtype=np.float32)

        def _cached_inference():
            key = info_vec.tobytes()
            return self._cached_policy.make(
                key, lambda: self._policy_network(jnp.asarray(info_vec))
            )

        probs = _cached_inference()
        legal_mask = jnp.asarray(state.legal_actions_mask(cur), dtype=jnp.bool)
        probs = jnp.where(legal_mask, probs, ILLEGAL_ACTION_LOGITS_PENALTY)
        probs = nn.softmax(probs)
        return {a: probs[a] for a in legal}

    def policy_for_exploitability(self) -> Callable[[str], np.ndarray]:
        """Build policy(infoset_key) -> [p_pass, p_bet] for exploitability."""

        def policy(infok: str) -> np.ndarray:
            obs = _infoset_to_obs(infok).astype(np.float32)
            key = obs.tobytes()
            logits = self._cached_policy.make(key, lambda: self._policy_network(jnp.asarray(obs)))
            logits = np.array(logits)
            probs = np.zeros(2, dtype=np.float64)
            probs[0] = np.exp(logits[0] - np.max(logits))
            probs[1] = np.exp(logits[1] - np.max(logits))
            probs /= probs.sum()
            return probs

        return policy

    def _learn_advantage_network(self, player: int) -> float | None:
        self._advantage_networks[player].train()
        buffer_state = self._advantage_memories[player]
        if buffer_state is None:
            return None
        batch_size = min(
            self._batch_size_advantage,
            len(buffer_state),
        )
        if batch_size <= 0:
            return None

        state = nn.state((self._advantage_networks[player], self._advantage_opt[player]))
        rng = self._next_rng_key()

        for _ in range(self._advantage_network_train_steps):
            rng, rng_ = jax.random.split(rng)
            batch = ReservoirBuffer.sample(rng_, buffer_state, batch_size)
            state, main_loss = self._jittable_adv_update(
                self._advantage_graphdefs[player], state, batch
            )

        nn.update(
            (self._advantage_networks[player], self._advantage_opt[player]),
            state,
        )
        return float(main_loss)

    def _learn_strategy_network(self) -> float | None:
        self._policy_network.train()
        if self._strategy_memories is None:
            return None
        batch_size = min(
            self._batch_size_strategy,
            len(self._strategy_memories),
        )
        if batch_size <= 0:
            return None

        state = nn.state((self._policy_network, self._policy_opt))
        rng = self._next_rng_key()

        for _ in range(self._advantage_network_train_steps):
            rng, rng_ = jax.random.split(rng)
            batch = ReservoirBuffer.sample(rng_, self._strategy_memories, batch_size)
            state, main_loss = self._jittable_policy_update(self._policy_graphdef, state, batch)

        nn.update((self._policy_network, self._policy_opt), state)
        return float(main_loss)


# --- Main entry ---
def run_deep_cfr(
    num_iterations: int = 50,
    num_traversals: int = 100,
    log_interval: int = 5,
    **kwargs,
) -> list[tuple[int, float]]:
    solver = DeepCFRSolver(
        policy_network_layers=(64, 64),
        advantage_network_layers=(64, 64),
        num_iterations=num_iterations,
        num_traversals=num_traversals,
        batch_size_advantage=min(256, 6 * num_traversals * 10),
        batch_size_strategy=min(256, 6 * num_traversals * 10),
        memory_capacity=100_000,
        advantage_network_train_steps=200,
        reinitialize_advantage_networks=True,
        seed=42,
        **kwargs,
    )

    root = _root_state()
    exploitability_log: list[tuple[int, float]] = []

    for it in range(num_iterations):
        if (it + 1) % log_interval == 0 or it == 0:
            solver._learn_strategy_network()
            pol = solver.policy_for_exploitability()
            expl = exploitability(pol)
            exploitability_log.append((solver._iteration, float(expl)))
            print(f"Iteration {solver._iteration:4d}: exploitability = {expl:.6f}")
            solver._reinitialize_policy_network()
            solver._cached_policy.clear()

        for p in range(2):
            for _ in range(num_traversals):
                solver._traverse_game_tree(root, p)
            if solver._reinitialize_advantage_networks:
                solver._reinitialize_advantage_network(p)
            solver._learn_advantage_network(p)

        solver._iteration += 1

    solver._learn_strategy_network()
    pol = solver.policy_for_exploitability()
    final_expl = exploitability(pol)
    exploitability_log.append((solver._iteration, float(final_expl)))
    print(f"Iteration {solver._iteration:4d}: exploitability = {final_expl:.6f} (final)")

    return exploitability_log


def main() -> None:
    print("Deep CFR for Kuhn Poker")
    print("=" * 50)
    expl_log = run_deep_cfr(
        num_iterations=100,
        num_traversals=10,
        log_interval=20,
    )
    print("=" * 50)
    print(f"Final exploitability: {expl_log[-1][1]:.6f}")


if __name__ == "__main__":
    main()
