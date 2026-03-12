"""
Game utilities for heads-up no-limit Texas Hold'em ReBeL.

Hand bucketing, PBS encoding, and subgame structure.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bluffjax.utils.typing import FloatArray, IntArray
from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
    TexasNoLimitHoldEmState,
)
from bluffjax.utils.game_utils.poker_utils import _card_rank, _card_suit

# Actions: 0=check/call, 1=raise_half_pot, 2=raise_pot, 3=all_in, 4=fold
NUM_ACTIONS = 5
NUM_AGENTS = 2
NUM_BUCKETS = 10
NUM_BETTING_ROUNDS = 4


def hand_strength_preflop(cards: IntArray) -> FloatArray:
    ranks = _card_rank(cards)
    suits = _card_suit(cards)
    high = jnp.max(ranks)
    low = jnp.min(ranks)
    is_pair = ranks[0] == ranks[1]
    is_suited = suits[0] == suits[1]
    pair_bonus = jnp.where(is_pair, 0.3, 0.0)
    high_bonus = (high - 2) / 13.0 * 0.5
    low_bonus = (low - 2) / 13.0 * 0.15
    suited_bonus = jnp.where(is_suited, 0.05, 0.0)
    strength = jnp.clip(pair_bonus + high_bonus + low_bonus + suited_bonus, 0.0, 1.0)
    return strength


def hand_to_bucket(cards: IntArray, board: IntArray, num_buckets: int) -> IntArray:
    del board
    strength = hand_strength_preflop(cards)
    bucket = (strength * (num_buckets - 1e-6)).astype(jnp.int32)
    return jnp.clip(bucket, 0, num_buckets - 1)


def encode_board(
    flop: FloatArray, turn: FloatArray, river: FloatArray, stage: IntArray
) -> FloatArray:
    board = jnp.zeros(52, dtype=jnp.float32)
    flop_0 = jnp.int32(flop[0])
    flop_1 = jnp.int32(flop[1])
    flop_2 = jnp.int32(flop[2])
    turn_i = jnp.int32(turn)
    river_i = jnp.int32(river)
    board = board.at[flop_0].set(jnp.where(stage >= 1, 1.0, 0.0))
    board = board.at[flop_1].set(jnp.where(stage >= 1, 1.0, 0.0))
    board = board.at[flop_2].set(jnp.where(stage >= 1, 1.0, 0.0))
    board = board.at[turn_i].set(jnp.where(stage >= 2, 1.0, 0.0))
    board = board.at[river_i].set(jnp.where(stage >= 3, 1.0, 0.0))
    return board


class HUNLPBSState(NamedTuple):
    stage: IntArray
    belief_p0: FloatArray  # (NUM_BUCKETS,)
    belief_p1: FloatArray  # (NUM_BUCKETS,)
    round_raised: FloatArray  # (2,)
    stack_frac: FloatArray  # (2,)
    board_enc: FloatArray  # (52,)
    position: IntArray  # 0=SB, 1=BB


def encode_pbs(pbs: HUNLPBSState, num_buckets: int) -> FloatArray:
    del num_buckets
    stage_oh = jax.nn.one_hot(pbs.stage, 4, dtype=jnp.float32)
    parts = [
        stage_oh,
        pbs.belief_p0,
        pbs.belief_p1,
        pbs.round_raised.astype(jnp.float32),
        pbs.stack_frac.astype(jnp.float32),
        pbs.board_enc,
        jax.nn.one_hot(pbs.position, 2, dtype=jnp.float32),
    ]
    return jnp.concatenate(parts, axis=-1)


def pbs_from_state(state: TexasNoLimitHoldEmState, num_buckets: int) -> HUNLPBSState:
    board_enc = encode_board(
        state.flop_cards,
        state.turn_card,
        state.river_card,
        state.stage,
    )
    position = (state.current_player_idx - state.small_blind_idx) % NUM_AGENTS
    uniform = jnp.ones(num_buckets, dtype=jnp.float32) / num_buckets
    chips_total = jnp.maximum(state.remaining_chips + state.chips_in, 1.0)
    stack_frac = state.remaining_chips / chips_total
    round_scale = jnp.maximum(jnp.max(state.round_raised), 1.0)
    round_raised_norm = state.round_raised / round_scale
    return HUNLPBSState(
        stage=state.stage,
        belief_p0=uniform,
        belief_p1=uniform,
        round_raised=round_raised_norm.astype(jnp.float32),
        stack_frac=stack_frac.astype(jnp.float32),
        board_enc=board_enc,
        position=position.astype(jnp.int32),
    )


PBS_INPUT_DIM = (
    4 + NUM_BUCKETS * 2 + 2 + 2 + 52 + 2
)  # stage + beliefs + round_raised + stack_frac + board + position
VALUE_OUTPUT_DIM = NUM_BUCKETS * 2
