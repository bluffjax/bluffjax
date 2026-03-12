"""
Game utilities for heads-up limit Texas hold'em ReBeL.

Hand bucketing, PBS encoding, and subgame structure.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bluffjax.utils.typing import FloatArray, IntArray
from bluffjax.environments.texas_limit_holdem.texas_limit_holdem import (
    TexasLimitHoldEmState,
)
from bluffjax.utils.game_utils.poker_utils import _card_rank, _card_suit

# Actions: 0=call, 1=raise, 2=fold, 3=check
NUM_ACTIONS = 4
NUM_AGENTS = 2
NUM_BUCKETS = 10
ALLOWED_RAISE_NUM = 4
NUM_BETTING_ROUNDS = 4


def hand_strength_preflop(cards: IntArray) -> FloatArray:
    """
    Compute preflop hand strength for bucketing.
    cards: (2,) hole card indices
    Returns scalar in [0, 1] for bucket assignment.
    """
    ranks = _card_rank(cards)
    suits = _card_suit(cards)
    high = jnp.max(ranks)
    low = jnp.min(ranks)
    is_pair = ranks[0] == ranks[1]
    # Simple strength: pair bonus + high card
    pair_bonus = jnp.where(is_pair, 0.3, 0.0)
    high_bonus = (high - 2) / 13.0 * 0.5
    low_bonus = (low - 2) / 13.0 * 0.2
    strength = jnp.clip(pair_bonus + high_bonus + low_bonus, 0.0, 1.0)
    return strength


def hand_to_bucket(cards: IntArray, board: IntArray, num_buckets: int) -> IntArray:
    """
    Map hole cards to bucket index given board (for postflop).
    Preflop: board is empty, use preflop strength.
    """
    strength = hand_strength_preflop(cards)
    bucket = (strength * (num_buckets - 1e-6)).astype(jnp.int32)
    return jnp.clip(bucket, 0, num_buckets - 1)


def encode_board(
    flop: FloatArray, turn: FloatArray, river: FloatArray, stage: IntArray
) -> FloatArray:
    """Encode board cards as 52-dim one-hot. Only set cards for revealed streets."""
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


class HULPBSState(NamedTuple):
    """Public belief state for HUL."""

    stage: IntArray
    belief_p0: FloatArray  # (NUM_BUCKETS,)
    belief_p1: FloatArray  # (NUM_BUCKETS,)
    raise_nums: FloatArray  # (4,) raise count per street
    board_enc: FloatArray  # (52,) one-hot board
    position: IntArray  # 0=SB, 1=BB (current player relative to small blind)


def encode_pbs(pbs: HULPBSState, num_buckets: int) -> FloatArray:
    """Encode PBS for value network input."""
    stage_oh = jax.nn.one_hot(pbs.stage, 4, dtype=jnp.float32)
    parts = [
        stage_oh,
        pbs.belief_p0,
        pbs.belief_p1,
        pbs.raise_nums.astype(jnp.float32) / ALLOWED_RAISE_NUM,
        pbs.board_enc,
        jax.nn.one_hot(pbs.position, 2, dtype=jnp.float32),
    ]
    return jnp.concatenate(parts, axis=-1)


def pbs_from_state(state: TexasLimitHoldEmState, num_buckets: int) -> HULPBSState:
    """Create PBS from env state. Uses uniform beliefs (placeholder for actual belief tracking)."""
    board_enc = encode_board(
        state.flop_cards,
        state.turn_card,
        state.river_card,
        state.stage,
    )
    position = (state.current_player_idx - state.small_blind_idx) % NUM_AGENTS
    uniform = jnp.ones(num_buckets, dtype=jnp.float32) / num_buckets
    return HULPBSState(
        stage=state.stage,
        belief_p0=uniform,
        belief_p1=uniform,
        raise_nums=state.raise_nums.astype(jnp.float32),
        board_enc=board_enc,
        position=position.astype(jnp.int32),
    )


PBS_INPUT_DIM = (
    4 + NUM_BUCKETS * 2 + 4 + 52 + 2
)  # stage + beliefs + raise_nums + board + position
VALUE_OUTPUT_DIM = NUM_BUCKETS * 2  # per-player values for each bucket
