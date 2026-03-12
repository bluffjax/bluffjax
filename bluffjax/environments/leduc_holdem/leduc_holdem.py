"""
Leduc Hold'em - OpenSpiel rules (matches new_test.py / new_cfr.py).

Exact port of OpenSpiel leduc_poker:
- 6-card deck: [J1, J2, Q1, Q2, K1, K2] encoded as [0..5]
- Ante 1 chip each, starting money 100
- Two betting rounds, max 2 raises per round
- Raise size 2 in round 1, 4 in round 2
- Actions: 0=Fold, 1=Call, 2=Raise (fold only when facing a bet)
- Returns = money - 100
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
from bluffjax.environments.env import AECEnv
from bluffjax.environments.spaces import Discrete

STARTING_MONEY = 100
ANTE = 1
RAISE_AMOUNT_R1 = 2
RAISE_AMOUNT_R2 = 4
MAX_RAISES = 2


def _rank_hand(
    agent_cards: IntArray, public_card: IntArray, player: IntArray
) -> IntArray:
    """Hand rank for comparison (matches new_test._rank_hand). Pair beats high card."""
    hand_0 = jnp.where(player == 0, public_card, agent_cards[1 - player])
    hand_1 = jnp.where(player == 0, agent_cards[player], public_card)
    lo = jnp.minimum(hand_0, hand_1)
    hi = jnp.maximum(hand_0, hand_1)
    # Pair: lo%2==0 and hi==lo+1
    is_pair = (lo % 2 == 0) & (hi == lo + 1)
    pair_rank = 6 * 6 + lo
    high_rank = (hi // 2) * 6 + (lo // 2)
    return jnp.where(is_pair, pair_rank, high_rank)


def _leduc_compare(
    agent_cards: IntArray, public_card: IntArray, folded: BoolArray
) -> FloatArray:
    """Determine winners (matches new_test hand ranking). Pair beats high card; ties split."""
    num_active = jnp.sum(~folded)
    single_winner = jnp.where(
        num_active == 1, (~folded).astype(jnp.float32), jnp.zeros(2, dtype=jnp.float32)
    )

    rank0 = _rank_hand(agent_cards, public_card, jnp.int32(0))
    rank1 = _rank_hand(agent_cards, public_card, jnp.int32(1))
    p0_wins = ((rank0 > rank1) & ~folded[0]).astype(jnp.float32)
    p1_wins = ((rank1 > rank0) & ~folded[1]).astype(jnp.float32)
    tie = (rank0 == rank1) & ~folded[0] & ~folded[1]

    w0 = jnp.where(tie, 0.5, single_winner[0] + p0_wins)
    w1 = jnp.where(tie, 0.5, single_winner[1] + p1_wins)
    winners = jnp.stack([w0, w1]).astype(jnp.float32)
    return winners


@struct.dataclass
class LeducHoldemState:
    agent_cards: IntArray  # (2,) physical cards 0-5
    public_card: IntArray  # -1 if round 1, else 0-5
    shuffled_deck: IntArray  # (6,)
    ante: FloatArray  # (2,) chips contributed per player
    stakes: FloatArray  # current bet level
    stage: IntArray  # 1 or 2 (round)
    num_calls: IntArray
    num_raises: IntArray
    folded: BoolArray
    current_player_idx: IntArray
    absorbing: BoolArray
    done: bool
    timestep: int


class LeducHoldem(AECEnv):
    """Leduc Hold'em: OpenSpiel rules (ante-based, 3 actions). Matches new_test.py exactly."""

    def __init__(self, num_agents: int = 2, horizon: int = 50) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.ante_amount = ANTE
        self.raise_amount_r1 = RAISE_AMOUNT_R1
        self.raise_amount_r2 = RAISE_AMOUNT_R2
        self.max_raises = MAX_RAISES
        self.starting_money = STARTING_MONEY
        self.obs_dim = 36
        self.num_actions = 3  # Fold, Call, Raise

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: LeducHoldemState) -> FloatArray:
        """36-dim: hand rank (0-2), public rank (3-5), my_ante (6-20), opp_ante (21-35). Matches new_test._infoset_key_to_obs."""
        current = state.current_player_idx
        opp = 1 - current
        hand = state.agent_cards[current]
        rank_hand = hand // 2  # physical 0-5 -> rank 0-2
        my_ante = jnp.minimum(state.ante[current].astype(jnp.int32), 14)
        opp_ante = jnp.minimum(state.ante[opp].astype(jnp.int32), 14)

        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        obs = obs.at[rank_hand].set(1.0)
        rank_public = jnp.where(state.public_card >= 0, state.public_card // 2, -1)
        obs = jnp.where(
            state.public_card >= 0,
            obs.at[3 + rank_public].set(1.0),
            obs,
        )
        obs = obs.at[6 + my_ante].set(1.0)
        obs = obs.at[21 + opp_ante].set(1.0)
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: LeducHoldemState) -> BoolArray:
        """Actions: 0=Fold, 1=Call, 2=Raise. Fold only when stakes > ante."""
        current = state.current_player_idx
        my_ante = state.ante[current]
        stakes = state.stakes
        num_raises = state.num_raises

        avail = jnp.zeros(3, dtype=bool)
        can_fold = stakes > my_ante
        avail = avail.at[0].set(can_fold)
        avail = avail.at[1].set(True)  # Call always
        can_raise = num_raises < self.max_raises
        avail = avail.at[2].set(can_raise)
        return avail

    def _ready_for_next_round(
        self, num_calls: IntArray, num_raises: IntArray, remaining: IntArray
    ) -> BoolArray:
        """OpenSpiel: (num_raises==0 && num_calls==remaining) || (num_raises>0 && num_calls==remaining-1)."""
        cond1 = (num_raises == 0) & (num_calls >= remaining)
        cond2 = (num_raises > 0) & (num_calls >= remaining - 1)
        return cond1 | cond2

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[LeducHoldemState, FloatArray]:
        """Deal cards, post ante. Player 0 starts round 1 (no blinds)."""
        rng_shuffle, _ = jax.random.split(rng)
        deck = jnp.arange(6, dtype=jnp.int32)  # physical cards 0-5
        shuffled_deck = jax.random.permutation(rng_shuffle, deck)
        agent_cards = shuffled_deck[:2]

        state = LeducHoldemState(
            agent_cards=agent_cards,
            public_card=jnp.int32(-1),
            shuffled_deck=shuffled_deck,
            ante=jnp.ones(2, dtype=jnp.float32) * self.ante_amount,
            stakes=jnp.float32(self.ante_amount),
            stage=jnp.int32(1),
            num_calls=jnp.int32(0),
            num_raises=jnp.int32(0),
            folded=jnp.zeros(2, dtype=bool),
            current_player_idx=jnp.int32(0),
            absorbing=jnp.zeros(2, dtype=bool),
            done=False,
            timestep=0,
        )
        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: LeducHoldemState,
        action: IntArray,
    ) -> tuple[
        LeducHoldemState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Execute one step. Actions: 0=Fold, 1=Call, 2=Raise."""
        current = state.current_player_idx
        my_ante = state.ante[current]
        stakes = state.stakes
        raise_amt = jnp.where(
            state.stage == 1, self.raise_amount_r1, self.raise_amount_r2
        )
        remaining = 2 - jnp.sum(state.folded)

        def process_fold():
            new_folded = state.folded.at[current].set(True)
            new_ante = state.ante
            new_stakes = state.stakes
            new_num_calls = state.num_calls
            new_num_raises = state.num_raises
            return new_ante, new_stakes, new_num_calls, new_num_raises, new_folded

        def process_call():
            amount = stakes - my_ante
            new_ante = state.ante.at[current].add(amount)
            new_num_calls = state.num_calls + 1
            return new_ante, state.stakes, new_num_calls, state.num_raises, state.folded

        def process_raise():
            call_amount = jnp.maximum(stakes - my_ante, 0.0)
            new_ante = state.ante.at[current].add(call_amount)
            new_stakes = stakes + raise_amt
            new_ante = new_ante.at[current].add(raise_amt)
            return (
                new_ante,
                new_stakes,
                jnp.int32(0),
                state.num_raises + 1,
                state.folded,
            )

        new_ante, new_stakes, new_num_calls, new_num_raises, new_folded = lax.switch(
            action, [process_fold, process_call, process_raise]
        )

        remaining_after = 2 - jnp.sum(new_folded)
        round_over = self._ready_for_next_round(
            new_num_calls, new_num_raises, remaining_after
        )

        round1_ended = (state.stage == 1) & round_over & (remaining_after > 1)
        public_card = jnp.where(round1_ended, state.shuffled_deck[2], state.public_card)
        new_stage = jnp.where(round1_ended, jnp.int32(2), state.stage)
        new_num_calls = jnp.where(round_over, jnp.int32(0), new_num_calls)
        new_num_raises = jnp.where(round_over, jnp.int32(0), new_num_raises)

        def next_player(start: IntArray, folded_mask: BoolArray) -> IntArray:
            other = 1 - start
            return jnp.where(folded_mask[other], start, other)

        next_after = next_player(current, new_folded)
        next_new_round = next_player(jnp.int32(0), new_folded)
        new_current = lax.select(round_over, next_new_round, next_after)

        game_done = (remaining_after <= 1) | ((state.stage == 2) & round_over)

        def compute_rewards() -> FloatArray:
            pot = jnp.sum(new_ante)
            winners = _leduc_compare(state.agent_cards, public_card, new_folded)
            winners = winners / jnp.maximum(jnp.sum(winners), 1e-8)
            chips_gained = winners * pot
            money = self.starting_money - new_ante
            final_money = money + chips_gained
            return (final_money - self.starting_money).astype(jnp.float32)

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(2, dtype=jnp.float32),
        )

        next_state = LeducHoldemState(
            agent_cards=state.agent_cards,
            public_card=public_card,
            shuffled_deck=state.shuffled_deck,
            ante=new_ante,
            stakes=new_stakes,
            stage=new_stage,
            num_calls=new_num_calls,
            num_raises=new_num_raises,
            folded=new_folded,
            current_player_idx=new_current,
            absorbing=jnp.broadcast_to(game_done, (2,)),
            done=game_done,
            timestep=state.timestep + 1,
        )
        obs = self.obs_from_state(next_state)
        absorbing = jnp.broadcast_to(game_done, (2,))
        info = {"timestep": next_state.timestep, "returns": rewards}
        return next_state, obs, rewards, absorbing, game_done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)
