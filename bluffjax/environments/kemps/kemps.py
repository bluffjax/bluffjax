"""
Kemps environment.

Simultaneous-move partnership card game. 4 players (2 teams of 2), each round
is one episode. Objective: get four-of-a-kind and have partner call KEMPS, or
call STOP KEMPS when opponent has four-of-a-kind.
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from functools import partial
from bluffjax.utils.typing import (
    Any,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)
from bluffjax.environments.env import ParallelEnv
from bluffjax.environments.spaces import Discrete


def _hand_to_binary(hand_cards: IntArray, deck_size: int) -> FloatArray:
    """Convert hand (card indices) to binary vector of shape (deck_size,)."""
    return (
        (jnp.arange(deck_size)[:, None] == hand_cards[None, :])
        .any(axis=1)
        .astype(jnp.float32)
    )


@struct.dataclass
class KempsState:
    """Kemps game state."""

    agent_hands: IntArray  # (num_agents, hand_size) - card indices
    agent_hand_counts: IntArray  # (num_agents, num_ranks)
    center_cards: IntArray  # (4,)
    center_counts: IntArray  # (num_ranks,)
    deck: IntArray  # remaining cards
    deck_idx: IntArray
    communication: FloatArray  # (num_agents, C) one-hot signals from last action
    absorbing: BoolArray
    done: bool
    timestep: int


class Kemps(ParallelEnv):
    """Kemps parallel environment. 2x2 teams, simultaneous moves."""

    def __init__(
        self,
        num_agents: int = 4,
        num_ranks: int = 13,
        hand_size: int = 4,
        num_suits: int = 4,
        comm_dim: int = 2,
        horizon: int = 200,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        assert num_agents % 2 == 0
        assert num_agents >= 4
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.num_suits = num_suits
        self.comm_dim = comm_dim
        self.deck_size = num_ranks * num_suits
        self.num_center = 4
        min_deck = num_agents * hand_size + self.num_center
        if self.deck_size < min_deck:
            raise ValueError(
                f"Deck too small: need {min_deck} cards, have {self.deck_size} "
                f"(num_ranks={num_ranks} * num_suits={num_suits})"
            )
        self.num_game_actions = (
            num_ranks * num_ranks + 3
        )  # swap, NOOP, KEMPS, STOP_KEMPS
        self.num_actions = (
            self.num_game_actions * comm_dim
        )  # each game action + comm signal
        self.obs_dim = (
            num_agents * self.deck_size + self.deck_size + num_agents * comm_dim
        )

    @partial(jax.jit, static_argnums=(0,))
    def _team_id(self, agent_idx: IntArray) -> IntArray:
        """Team 0: agents 0,2; Team 1: agents 1,3."""
        return agent_idx % 2

    @partial(jax.jit, static_argnums=(0,))
    def _partner_idx(self, agent_idx: IntArray) -> IntArray:
        """Partner sits across: 0<->2, 1<->3."""
        return (agent_idx + 2) % self.num_agents

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: KempsState) -> FloatArray:
        """
        Observation per agent: (num_agents, obs_dim).
        - Agent cards: (num_agents, deck_size) binary, own row filled only
        - Public cards: (deck_size,) binary
        - Communication: (num_agents, C) one-hot from _rel_array
        """

        # Agent cards: for each agent i, row i = own hand binary, others = 0
        def agent_cards_for(i):
            hand = state.agent_hands[i]
            binary = _hand_to_binary(hand, self.deck_size)
            # Mask: only row i has real data
            all_rows = jnp.zeros((self.num_agents, self.deck_size), dtype=jnp.float32)
            all_rows = all_rows.at[i].set(binary)
            return all_rows

        agent_cards = jax.vmap(agent_cards_for)(self.agent_idxs)
        agent_cards_flat = agent_cards.reshape(self.num_agents, -1)

        # Public cards binary
        public_binary = (
            (jnp.arange(self.deck_size)[:, None] == state.center_cards[None, :])
            .any(axis=1)
            .astype(jnp.float32)
        )
        public_tile = jnp.broadcast_to(public_binary, (self.num_agents, self.deck_size))

        # Communication (relative) - (num_agents, num_agents, C) -> (num_agents, num_agents*C)
        comm_rel = self._rel_array(state.communication)
        comm_rel_flat = comm_rel.reshape(self.num_agents, -1)

        obs = jnp.concatenate(
            [agent_cards_flat, public_tile, comm_rel_flat],
            axis=1,
        )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: KempsState) -> BoolArray:
        """(num_agents, num_actions) - each game action valid with any comm signal."""
        swap_valid = jnp.zeros(
            (self.num_agents, self.num_ranks * self.num_ranks), dtype=bool
        )

        def check_swap(agent_idx):
            hand_counts = state.agent_hand_counts[agent_idx]
            center_counts = state.center_counts
            has_lose = hand_counts[:, None] >= 1
            center_has_gain = center_counts[None, :] >= 1
            valid = has_lose & center_has_gain
            return valid.ravel()

        swap_valid = jax.vmap(check_swap)(self.agent_idxs)

        noop_valid = jnp.ones((self.num_agents, 1), dtype=bool)
        declare_valid = jnp.ones((self.num_agents, 2), dtype=bool)

        game_avail = jnp.concatenate([swap_valid, noop_valid, declare_valid], axis=1)
        # Expand: each game action pairs with comm_dim signals -> (num_agents, num_game_actions * comm_dim)
        return jnp.repeat(game_avail, self.comm_dim, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[KempsState, FloatArray]:
        """Shuffle deck, deal hand_size per player, 4 to center."""
        rng_shuffle, rng_deal = jax.random.split(rng)
        deck = jax.random.permutation(rng_shuffle, self.deck_size)

        cards_per_player = self.hand_size * self.num_agents
        center_start = cards_per_player
        deck_used = center_start + self.num_center

        agent_hands = deck[:cards_per_player].reshape(self.num_agents, self.hand_size)
        center_cards = deck[center_start : center_start + self.num_center]
        remaining_deck = deck[deck_used:]

        agent_hand_counts = jnp.zeros(
            (self.num_agents, self.num_ranks), dtype=jnp.int32
        )

        def count_ranks(hand):
            ranks = hand // self.num_suits
            return jnp.bincount(ranks, length=self.num_ranks, minlength=self.num_ranks)

        agent_hand_counts = jax.vmap(count_ranks)(agent_hands)

        center_ranks = center_cards // self.num_suits
        center_counts = jnp.bincount(
            center_ranks.astype(jnp.int32),
            length=self.num_ranks,
            minlength=self.num_ranks,
        )

        communication = jnp.zeros(
            (self.num_agents, self.comm_dim), dtype=jnp.float32
        )  # one-hot, zeros = no signal

        state = KempsState(
            agent_hands=agent_hands,
            agent_hand_counts=agent_hand_counts,
            center_cards=center_cards,
            center_counts=center_counts,
            deck=remaining_deck,
            deck_idx=jnp.int32(0),
            communication=communication,
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=False,
            timestep=0,
        )
        obs = self.obs_from_state(state)
        return state, obs

    def _process_swaps(
        self,
        rng: PRNGKeyArray,
        state: KempsState,
        game_action: IntArray,
    ) -> KempsState:
        """Process swap actions in random order, with tie-breaking."""
        swap_action = num_ranks_sq = self.num_ranks * self.num_ranks
        is_swap = game_action < swap_action

        rng_perm, _ = jax.random.split(rng)
        order = jax.random.permutation(rng_perm, self.num_agents)

        def scan_fn(carry, i):
            hands, hand_counts, center_cards, center_counts = carry
            agent_idx = order[i]
            action = game_action[agent_idx]
            doing_swap = is_swap[agent_idx] & (action < swap_action)

            lose_rank = action // self.num_ranks
            gain_rank = action % self.num_ranks

            has_lose = hand_counts[agent_idx, lose_rank] >= 1
            center_has_gain = center_counts[gain_rank] >= 1
            valid = doing_swap & has_lose & center_has_gain

            # Find first card of rank lose in hand
            hand = hands[agent_idx]
            rank_of_cards = hand // self.num_suits
            lose_mask = (rank_of_cards == lose_rank).astype(jnp.int32)
            discard_pos = jnp.argmax(lose_mask)

            # Find first card of rank gain in center
            center_ranks = center_cards // self.num_suits
            gain_mask = (center_ranks == gain_rank).astype(jnp.int32)
            take_pos = jnp.argmax(gain_mask)

            discard_card = hand[discard_pos]
            take_card = center_cards[take_pos]

            def do_swap(_):
                new_hand = hand.at[discard_pos].set(take_card)
                new_hands = hands.at[agent_idx].set(new_hand)
                new_center = center_cards.at[take_pos].set(discard_card)
                new_hand_counts = hand_counts.at[agent_idx, lose_rank].add(-1)
                new_hand_counts = new_hand_counts.at[agent_idx, gain_rank].add(1)
                new_center_counts = center_counts.at[gain_rank].add(-1)
                new_center_counts = new_center_counts.at[lose_rank].add(1)
                return new_hands, new_hand_counts, new_center, new_center_counts

            def no_swap(_):
                return hands, hand_counts, center_cards, center_counts

            new_hands, new_hand_counts, new_center, new_center_counts = lax.cond(
                valid, do_swap, no_swap, None
            )
            return (new_hands, new_hand_counts, new_center, new_center_counts), None

        (hands, hand_counts, center_cards, center_counts), _ = lax.scan(
            scan_fn,
            (
                state.agent_hands,
                state.agent_hand_counts,
                state.center_cards,
                state.center_counts,
            ),
            jnp.arange(self.num_agents),
        )

        return state.replace(
            agent_hands=hands,
            agent_hand_counts=hand_counts,
            center_cards=center_cards,
            center_counts=center_counts,
        )

    def _refresh_center(self, rng: PRNGKeyArray, state: KempsState) -> KempsState:
        """Sweep center, deal 4 new cards from deck."""
        deck_len = state.deck.shape[0]
        can_deal = state.deck_idx + self.num_center <= deck_len

        new_center = lax.cond(
            can_deal,
            lambda: lax.dynamic_slice(
                state.deck, (state.deck_idx,), (self.num_center,)
            ),
            lambda: state.center_cards,
        )
        new_center_counts = jnp.where(
            can_deal,
            jnp.bincount(
                (new_center // self.num_suits).astype(jnp.int32),
                length=self.num_ranks,
                minlength=self.num_ranks,
            ),
            state.center_counts,
        )
        new_deck_idx = jnp.where(
            can_deal, state.deck_idx + self.num_center, state.deck_idx
        )

        return state.replace(
            center_cards=new_center,
            center_counts=new_center_counts,
            deck_idx=new_deck_idx,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: KempsState,
        action: IntArray,
    ) -> tuple[
        KempsState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Execute one step. action: (num_agents,) - game_action * comm_dim + comm_signal."""
        game_action = action // self.comm_dim
        comm_signal = action % self.comm_dim
        communication = jax.nn.one_hot(comm_signal, self.comm_dim, dtype=jnp.float32)

        num_ranks_sq = self.num_ranks * self.num_ranks
        action_noop = num_ranks_sq
        action_kemps = num_ranks_sq + 1
        action_stop_kemps = num_ranks_sq + 2

        declared_kemps = (game_action == action_kemps).any()
        declared_stop = (game_action == action_stop_kemps).any()
        any_declare = declared_kemps | declared_stop

        # Resolve declaration: first declarer wins (lowest index)
        def first_kemps_idx():
            mask = game_action == action_kemps
            idxs = jnp.where(mask, self.agent_idxs, self.num_agents)
            return jnp.min(idxs)

        def first_stop_idx():
            mask = game_action == action_stop_kemps
            idxs = jnp.where(mask, self.agent_idxs, self.num_agents)
            return jnp.min(idxs)

        kemps_caller = first_kemps_idx()
        stop_caller = first_stop_idx()

        # KEMPS takes precedence if both declared (arbitrary but consistent)
        kemps_first = declared_kemps & (~declared_stop | (kemps_caller < stop_caller))

        def resolve_kemps():
            partner = self._partner_idx(kemps_caller)
            partner_has_4 = (state.agent_hand_counts[partner] == 4).any()
            caller_team = self._team_id(kemps_caller)
            # If partner has 4: opposing team loses (we win) -> +1 for us
            # Else: we lose -> -1 for us
            win = partner_has_4
            rewards = jnp.where(
                self._team_id(self.agent_idxs) == caller_team,
                jnp.where(win, 1.0, -1.0),
                jnp.where(win, -1.0, 1.0),
            ).astype(jnp.float32)
            return rewards

        def resolve_stop():
            # Opponents are the other team (the one we're accusing)
            opp_team = 1 - self._team_id(stop_caller)
            opp_mask = self._team_id(self.agent_idxs) == opp_team
            opp_has_4 = ((state.agent_hand_counts == 4).any(axis=1) & opp_mask).any()
            caller_team = self._team_id(stop_caller)
            # If opponent has 4: they lose (we win) -> +1 for us
            # Else: we lose -> -1 for us
            win = opp_has_4
            rewards = jnp.where(
                self._team_id(self.agent_idxs) == caller_team,
                jnp.where(win, 1.0, -1.0),
                jnp.where(win, -1.0, 1.0),
            ).astype(jnp.float32)
            return rewards

        declare_rewards = lax.cond(kemps_first, resolve_kemps, resolve_stop)

        # No declaration: process swaps or refresh center
        all_noop = (game_action == action_noop).all()

        rng_swap, rng_refresh, _ = jax.random.split(rng, 3)

        state_after_swap = lax.cond(
            ~any_declare & ~all_noop,
            lambda: self._process_swaps(rng_swap, state, game_action),
            lambda: state,
        )

        deck_len = state.deck.shape[0]
        deck_exhausted = state.deck_idx + self.num_center > deck_len

        state_after_refresh = lax.cond(
            ~any_declare & all_noop,
            lambda: self._refresh_center(rng_refresh, state),
            lambda: state_after_swap,
        )

        real_deal = ~any_declare & all_noop & deck_exhausted
        real_deal_rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        rewards = lax.cond(
            any_declare,
            lambda: declare_rewards,
            lambda: lax.cond(
                real_deal,
                lambda: real_deal_rewards,
                lambda: jnp.zeros(self.num_agents, dtype=jnp.float32),
            ),
        )

        done = any_declare | real_deal
        absorbing = jnp.broadcast_to(done, (self.num_agents,))

        new_communication = lax.cond(
            ~done,
            lambda: communication,
            lambda: state.communication,
        )

        next_state = state_after_refresh.replace(
            communication=new_communication,
            absorbing=absorbing,
            done=done,
            timestep=state.timestep + 1,
        )

        # Horizon truncation
        horizon_done = state.timestep + 1 >= self.horizon
        next_state = next_state.replace(done=next_state.done | horizon_done)
        absorbing = absorbing | horizon_done

        obs = self.obs_from_state(next_state)
        return next_state, obs, rewards, absorbing, next_state.done, {}

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: PRNGKeyArray,
        state: KempsState,
        action: IntArray,
    ) -> tuple[
        KempsState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Step with unified action (num_agents,); reset on done."""
        rng_step, rng_reset = jax.random.split(rng)
        state_next, obs, rewards, absorbing, done, info = self.step_env(
            rng_step, state, action
        )
        state_reset, obs_reset = self.reset(rng_reset)
        state_final = lax.cond(done, lambda: state_reset, lambda: state_next)
        obs_final = lax.cond(done, lambda: obs_reset, lambda: obs)
        return state_final, obs_final, rewards, absorbing, done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)
