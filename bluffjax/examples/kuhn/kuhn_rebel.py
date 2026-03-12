"""
ReBeL (Recursive Belief-based Learning) for Kuhn Poker.

Full pipeline: depth-limited subgames, CFR-D, value network, self-play.
Training loop is fully jittable.

Combined from kuhn_rebel_tree, kuhn_rebel_core, and main training loop.
"""

import datetime
import itertools
from typing import Callable, NamedTuple

import flax.linen as nn
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
from bluffjax.utils.game_utils.kuhn_exploitability import (
    ACTION_BET,
    ACTION_PASS,
    exploitability,
    get_current_player,
    get_legal_actions,
    get_returns,
    infoset_key,
    is_terminal,
)
from bluffjax.utils.wandb_multilogger import WandbMultiLogger


# =============================================================================
# Tree structure
# =============================================================================

NUM_DEALS = 6
NUM_HISTORIES = 9
NUM_INFOSETS = 12
NUM_ACTIONS = 2
NUM_PUBLIC_STATES = 4  # "", "p", "b", "pb" (in subgame with max_depth=2)

HISTORIES: tuple[tuple[int, ...], ...] = (
    (),
    (ACTION_PASS,),
    (ACTION_BET,),
    (ACTION_PASS, ACTION_PASS),
    (ACTION_PASS, ACTION_BET),
    (ACTION_BET, ACTION_PASS),
    (ACTION_BET, ACTION_BET),
    (ACTION_PASS, ACTION_BET, ACTION_PASS),
    (ACTION_PASS, ACTION_BET, ACTION_BET),
)

HISTORY_TO_IDX: dict[tuple[int, ...], int] = {h: i for i, h in enumerate(HISTORIES)}

INFOSET_KEYS: tuple[str, ...] = (
    "0",
    "1",
    "2",
    "0p",
    "1p",
    "2p",
    "0b",
    "1b",
    "2b",
    "0pb",
    "1pb",
    "2pb",
)

INFOSET_KEY_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(INFOSET_KEYS)}

DEALS: np.ndarray = np.array(
    list(itertools.permutations([0, 1, 2], 2)),
    dtype=np.int32,
)

PUBLIC_STATES: tuple[str, ...] = ("", "p", "b", "pb")
PUBLIC_STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(PUBLIC_STATES)}

IS_TERMINAL_ARR: np.ndarray = np.array(
    [is_terminal(HISTORIES[i]) for i in range(NUM_HISTORIES)],
    dtype=np.bool_,
)


def _depth(hist: tuple[int, ...]) -> int:
    return len(hist)


IS_LEAF_ARR: np.ndarray = np.array(
    [
        _depth(HISTORIES[i]) == 2 and not is_terminal(HISTORIES[i])
        for i in range(NUM_HISTORIES)
    ],
    dtype=np.bool_,
)

CURRENT_PLAYER_ARR: np.ndarray = np.array(
    [get_current_player(HISTORIES[i]) for i in range(NUM_HISTORIES)],
    dtype=np.int32,
)

LEGAL_ACTIONS_ARR: np.ndarray = np.zeros((NUM_HISTORIES, NUM_ACTIONS), dtype=np.bool_)
for i in range(NUM_HISTORIES):
    legal = get_legal_actions(HISTORIES[i])
    for a in legal:
        LEGAL_ACTIONS_ARR[i, a] = True

LEGAL_ACTIONS_INFOSET_ARR: np.ndarray = np.ones(
    (NUM_INFOSETS, NUM_ACTIONS), dtype=np.bool_
)

CHILD_HIST_IDX: np.ndarray = np.full((NUM_HISTORIES, NUM_ACTIONS), -1, dtype=np.int32)
for i in range(NUM_HISTORIES):
    hist = HISTORIES[i]
    if is_terminal(hist):
        continue
    for a in get_legal_actions(hist):
        child = hist + (a,)
        if child in HISTORY_TO_IDX:
            CHILD_HIST_IDX[i, a] = HISTORY_TO_IDX[child]


def _infoset_idx_for_node(deal_idx: int, hist_idx: int) -> int:
    hands = tuple(DEALS[deal_idx])
    hist = HISTORIES[hist_idx]
    if is_terminal(hist):
        return -1
    cur = get_current_player(hist)
    card = hands[cur]
    key = infoset_key(card, hist)
    return INFOSET_KEY_TO_IDX[key]


INFOSET_AT_NODE: np.ndarray = np.zeros((NUM_DEALS, NUM_HISTORIES), dtype=np.int32)
for d in range(NUM_DEALS):
    for h in range(NUM_HISTORIES):
        INFOSET_AT_NODE[d, h] = _infoset_idx_for_node(d, h)

RETURNS_ARR: np.ndarray = np.zeros((NUM_DEALS, NUM_HISTORIES, 2), dtype=np.float32)
for d in range(NUM_DEALS):
    for h in range(NUM_HISTORIES):
        if IS_TERMINAL_ARR[h]:
            r0, r1 = get_returns(tuple(DEALS[d]), HISTORIES[h])
            RETURNS_ARR[d, h, 0] = r0
            RETURNS_ARR[d, h, 1] = r1

DEPTH_ORDER: list[int] = sorted(
    range(NUM_HISTORIES), key=lambda i: _depth(HISTORIES[i])
)
REVERSE_DEPTH_ORDER: list[int] = list(reversed(DEPTH_ORDER))

# Parent history index and action from parent (for reach computation)
PARENT_HIST_IDX: np.ndarray = np.full(NUM_HISTORIES, -1, dtype=np.int32)
ACTION_FROM_PARENT: np.ndarray = np.full(NUM_HISTORIES, -1, dtype=np.int32)
for h in range(1, NUM_HISTORIES):
    parent_hist = HISTORIES[h][:-1]
    PARENT_HIST_IDX[h] = HISTORY_TO_IDX[parent_hist]
    ACTION_FROM_PARENT[h] = HISTORIES[h][-1]


# Subtree: IN_SUBTREE[root, h] = True if h is root or a descendant of root
def _is_descendant(h: int, root: int) -> bool:
    hr, hh = HISTORIES[root], HISTORIES[h]
    if len(hh) < len(hr):
        return False
    return hh[: len(hr)] == hr


IN_SUBTREE: np.ndarray = np.zeros((NUM_HISTORIES, NUM_HISTORIES), dtype=np.bool_)
for root in range(NUM_HISTORIES):
    for h in range(NUM_HISTORIES):
        IN_SUBTREE[root, h] = _is_descendant(h, root)


def get_jax_arrays() -> dict[str, jnp.ndarray]:
    """Return dict of JAX arrays for tree structure."""
    return {
        "is_terminal": jnp.array(IS_TERMINAL_ARR),
        "is_leaf": jnp.array(IS_LEAF_ARR),
        "current_player": jnp.array(CURRENT_PLAYER_ARR),
        "legal_actions": jnp.array(LEGAL_ACTIONS_ARR),
        "legal_actions_infoset": jnp.array(LEGAL_ACTIONS_INFOSET_ARR),
        "child_hist_idx": jnp.array(CHILD_HIST_IDX),
        "infoset_at_node": jnp.array(INFOSET_AT_NODE),
        "returns": jnp.array(RETURNS_ARR),
        "deals": jnp.array(DEALS),
        "reverse_depth_order": jnp.array(REVERSE_DEPTH_ORDER),
        "parent_hist_idx": jnp.array(PARENT_HIST_IDX),
        "action_from_parent": jnp.array(ACTION_FROM_PARENT),
        "in_subtree": jnp.array(IN_SUBTREE),
    }


# =============================================================================
# PBS, value network, CFR-D
# =============================================================================

PBS_INPUT_DIM = 11
VALUE_OUTPUT_DIM = 6


class PBSState(NamedTuple):
    """Public belief state: public_state_idx, beliefs for each player."""

    public_state_idx: IntArray
    belief_p0: FloatArray
    belief_p1: FloatArray


def decode_pbs(enc: FloatArray) -> PBSState:
    """Decode PBS from flat vector. enc shape (PBS_INPUT_DIM,)."""
    pub_idx = jnp.argmax(enc[0:NUM_PUBLIC_STATES])
    return PBSState(
        public_state_idx=pub_idx.astype(jnp.int32),
        belief_p0=enc[NUM_PUBLIC_STATES : NUM_PUBLIC_STATES + 3],
        belief_p1=enc[NUM_PUBLIC_STATES + 3 : NUM_PUBLIC_STATES + 6],
    )


def encode_pbs(pbs: PBSState) -> FloatArray:
    """Encode PBS as flat vector for value network input. Shape (PBS_INPUT_DIM,)."""
    pub_oh = jax.nn.one_hot(pbs.public_state_idx, NUM_PUBLIC_STATES, dtype=jnp.float32)
    acting = jnp.where(
        (pbs.public_state_idx == 0) | (pbs.public_state_idx == 3),
        jnp.float32(0.0),
        jnp.float32(1.0),
    ).reshape(1)
    return jnp.concatenate([pub_oh, pbs.belief_p0, pbs.belief_p1, acting], axis=-1)


def root_pbs() -> PBSState:
    """Initial PBS at game root: uniform beliefs."""
    return PBSState(
        public_state_idx=jnp.int32(0),
        belief_p0=jnp.ones(3, dtype=jnp.float32) / 3.0,
        belief_p1=jnp.ones(3, dtype=jnp.float32) / 3.0,
    )


def update_belief_after_action(
    pbs: PBSState,
    action: IntArray,
    policy: FloatArray,
) -> PBSState:
    """Update belief when transitioning from parent to child via action."""
    pub_idx = pbs.public_state_idx

    def from_root():
        def pass_action():
            probs = policy[0:3, 0]
            belief_p0_new = probs / (jnp.sum(probs) + 1e-8)
            return PBSState(
                public_state_idx=jnp.int32(1),
                belief_p0=belief_p0_new,
                belief_p1=jnp.ones(3, dtype=jnp.float32) / 3.0,
            )

        def bet_action():
            probs = policy[0:3, 1]
            belief_p0_new = probs / (jnp.sum(probs) + 1e-8)
            return PBSState(
                public_state_idx=jnp.int32(2),
                belief_p0=belief_p0_new,
                belief_p1=jnp.ones(3, dtype=jnp.float32) / 3.0,
            )

        return jax.lax.cond(action == 0, pass_action, bet_action)

    def from_p():
        def bet_action():
            probs = policy[3:6, 1]
            belief_p1_new = probs / (jnp.sum(probs) + 1e-8)
            return PBSState(
                public_state_idx=jnp.int32(3),
                belief_p0=pbs.belief_p0,
                belief_p1=belief_p1_new,
            )

        return jax.lax.cond(action == 1, bet_action, lambda: pbs)

    return jax.lax.cond(
        pub_idx == 0,
        from_root,
        lambda: jax.lax.cond(pub_idx == 1, from_p, lambda: pbs),
    )


class ValueNetworkMLP(nn.Module):
    """MLP that maps PBS encoding to 6 infostate values (3 per player)."""

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


def _get_tree_arrays() -> dict:
    arrs = get_jax_arrays()
    arrs["deals"] = jnp.array(DEALS)
    arrs["returns"] = jnp.array(RETURNS_ARR)
    return arrs


# Public state index -> history index: ""->0, "p"->1, "b"->2, "pb"->4
PUBLIC_STATE_TO_HIST_IDX: tuple[int, ...] = (0, 1, 2, 4)


def _values_at_pbs_to_target(values: FloatArray, pub_state_idx: IntArray) -> FloatArray:
    """
    Extract 6-dim value target for a PBS from the CFR values array.
    values: (NUM_DEALS, NUM_HISTORIES, 2)
    Returns [P0_card0, P0_card1, P0_card2, P1_card0, P1_card1, P1_card2].
    """
    deals = jnp.array(DEALS)
    hist_idx = jnp.take(
        jnp.array(PUBLIC_STATE_TO_HIST_IDX),
        jnp.int32(pub_state_idx),
    )
    v_p0 = values[:, hist_idx, 0]  # (6,)
    v_p1 = values[:, hist_idx, 1]  # (6,)
    cards_p0 = deals[:, 0]
    cards_p1 = deals[:, 1]

    # For each card c in {0,1,2}, average over deals where that player has card c
    # Kuhn: each card appears exactly twice per player
    def avg_p0(c):
        mask = (cards_p0 == c).astype(jnp.float32)
        return jnp.sum(v_p0 * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def avg_p1(c):
        mask = (cards_p1 == c).astype(jnp.float32)
        return jnp.sum(v_p1 * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    return jnp.array(
        [
            avg_p0(0),
            avg_p0(1),
            avg_p0(2),
            avg_p1(0),
            avg_p1(1),
            avg_p1(2),
        ],
        dtype=jnp.float32,
    )


def cfr_d_one_iteration(
    cumulative_regret: FloatArray,
    cumulative_policy: FloatArray,
    root_hist_idx: IntArray,
    iteration: int,
    linear_averaging: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """One CFR-D iteration. Returns (new_regret, new_cum_policy, root_values, values)."""
    tree = _get_tree_arrays()
    in_subtree = tree["in_subtree"]
    legal_infoset = tree["legal_actions_infoset"]
    is_terminal_arr = tree["is_terminal"]
    current_player = tree["current_player"]
    child_hist_idx = tree["child_hist_idx"]
    infoset_at_node = tree["infoset_at_node"]
    returns_arr = tree["returns"]
    deals = tree["deals"]

    policy = jax.vmap(
        lambda r, m: regret_matching(r, m),
        in_axes=(0, 0),
    )(cumulative_regret, legal_infoset)

    values = jnp.zeros((NUM_DEALS, NUM_HISTORIES, 2), dtype=jnp.float32)

    def process_hist(values, h):
        def term_case():
            return values.at[:, h, :].set(returns_arr[:, h, :])

        def internal_case():
            # Use terminal returns from children (no value network at leaf)
            def deal_body(d):
                infoset = infoset_at_node[d, h]
                ch = child_hist_idx[h, :]
                ch_safe = jnp.maximum(ch, 0)
                contrib0 = policy[infoset, :] * values[d, ch_safe, 0]
                contrib1 = policy[infoset, :] * values[d, ch_safe, 1]
                contrib0 = jnp.where(ch >= 0, contrib0, 0.0)
                contrib1 = jnp.where(ch >= 0, contrib1, 0.0)
                return jnp.sum(contrib0), jnp.sum(contrib1)

            v0_arr, v1_arr = jax.vmap(deal_body)(jnp.arange(NUM_DEALS))
            return values.at[:, h, 0].set(v0_arr).at[:, h, 1].set(v1_arr)

        return jax.lax.cond(
            is_terminal_arr[h],
            term_case,
            internal_case,
        )

    for h in REVERSE_DEPTH_ORDER:
        values = process_hist(values, h)

    # Compute reach probabilities for subgame rooted at root_hist_idx
    parent_hist_idx_arr = tree["parent_hist_idx"]
    action_from_parent_arr = tree["action_from_parent"]
    reach_p0 = jnp.zeros((NUM_DEALS, NUM_HISTORIES), dtype=jnp.float32)
    reach_p1 = jnp.zeros((NUM_DEALS, NUM_HISTORIES), dtype=jnp.float32)
    reach_p0 = reach_p0.at[:, root_hist_idx].set(1.0)
    reach_p1 = reach_p1.at[:, root_hist_idx].set(1.0)
    for h in DEPTH_ORDER:
        parent = parent_hist_idx_arr[h]
        is_root = (h == root_hist_idx) | (parent < 0)
        parent_in_subtree = jnp.where(
            parent >= 0, in_subtree[root_hist_idx, parent], False
        )
        action = action_from_parent_arr[h]
        parent_acting = current_player[parent]
        infosets_at_parent = infoset_at_node[:, parent]
        policy_probs = jnp.take(policy[:, action], infosets_at_parent)
        r0_new = jnp.where(
            parent_acting == 0,
            reach_p0[:, parent] * policy_probs,
            reach_p0[:, parent],
        )
        r1_new = jnp.where(
            parent_acting == 1,
            reach_p1[:, parent] * policy_probs,
            reach_p1[:, parent],
        )
        use_parent = parent_in_subtree.astype(jnp.float32)
        in_subtree_h = in_subtree[root_hist_idx, h].astype(jnp.float32)
        r0_val = jnp.where(is_root, 1.0, use_parent * r0_new + (1 - use_parent) * 1.0)
        r1_val = jnp.where(is_root, 1.0, use_parent * r1_new + (1 - use_parent) * 1.0)
        reach_p0 = reach_p0.at[:, h].set(in_subtree_h * r0_val)
        reach_p1 = reach_p1.at[:, h].set(in_subtree_h * r1_val)

    def update_regrets_for_node(cr, cp, d, h):
        in_sub = in_subtree[root_hist_idx, h]

        def do_update():
            cur = current_player[h]
            infoset = infoset_at_node[d, h]
            # CF reach: opponent's reach (matches kuhn_cfr.py)
            opp_reach = jnp.where(cur == 0, reach_p1[d, h], reach_p0[d, h])
            cf_reach = opp_reach
            self_reach = jnp.where(cur == 0, reach_p0[d, h], reach_p1[d, h])
            ch0, ch1 = child_hist_idx[h, 0], child_hist_idx[h, 1]
            child_v0 = jnp.where(ch0 >= 0, values[d, ch0, cur], 0.0)
            child_v1 = jnp.where(ch1 >= 0, values[d, ch1, cur], 0.0)
            node_v = values[d, h, cur]
            regret0 = jnp.where(
                jnp.logical_and(legal_infoset[infoset, 0], ch0 >= 0),
                cf_reach * (child_v0 - node_v),
                0.0,
            )
            regret1 = jnp.where(
                jnp.logical_and(legal_infoset[infoset, 1], ch1 >= 0),
                cf_reach * (child_v1 - node_v),
                0.0,
            )
            new_cr = cr.at[infoset, 0].add(regret0).at[infoset, 1].add(regret1)
            w = jnp.where(
                linear_averaging,
                iteration * self_reach,
                self_reach,
            )
            new_cp = (
                cp.at[infoset, 0]
                .add(w * policy[infoset, 0])
                .at[infoset, 1]
                .add(w * policy[infoset, 1])
            )
            return new_cr, new_cp

        return jax.lax.cond(
            jnp.logical_or(is_terminal_arr[h], jnp.logical_not(in_sub)),
            lambda: (cr, cp),
            do_update,
        )

    new_regret = cumulative_regret
    new_cum_pol = cumulative_policy
    for d in range(NUM_DEALS):
        for h in range(NUM_HISTORIES):
            new_regret, new_cum_pol = update_regrets_for_node(
                new_regret, new_cum_pol, d, h
            )

    root_v0 = jnp.mean(values[:, root_hist_idx, 0])
    root_v1 = jnp.mean(values[:, root_hist_idx, 1])
    root_values = jnp.array([root_v0, root_v1], dtype=jnp.float32)
    return new_regret, new_cum_pol, root_values, values


def get_average_policy_from_cfr(
    cumulative_policy: FloatArray,
    legal: FloatArray,
) -> FloatArray:
    """Convert cumulative policy to average policy. Shape (NUM_INFOSETS, NUM_ACTIONS)."""
    total = jnp.sum(cumulative_policy, axis=-1, keepdims=True)
    avg = jnp.where(total > 1e-8, cumulative_policy / total, 0.5)
    return jnp.where(legal, avg, 0.0)


# Precomputed: for each infoset, expected terminal return for each action.
# Action leads to terminal: use this value. Else: use value network.
# (infoset_idx, action) -> (is_terminal, value_if_terminal)
# Deal indices per infoset: acting player has card c -> deals where that player has c
_INFOSET_DEAL_IDXS: list[tuple[int, ...]] = [
    (0, 1),  # "0": P0 card 0 -> (0,1),(0,2)
    (2, 3),  # "1": P0 card 1
    (4, 5),  # "2": P0 card 2
    (2, 4),  # "0p": P1 card 0
    (0, 5),  # "1p": P1 card 1
    (1, 3),  # "2p": P1 card 2
    (2, 4),  # "0b"
    (0, 5),  # "1b"
    (1, 3),  # "2b"
    (0, 1),  # "0pb": P0 card 0
    (2, 3),  # "1pb"
    (4, 5),  # "2pb"
]
# Terminal history index per (infoset, action). -1 means use value network.
# Histories: pp=3, bp=4, bb=5, pbp=7, pbb=8
_INFOSET_ACTION_TO_TERM_HIST: list[tuple[int, int]] = [
    (-1, -1),  # "0": both use network
    (-1, -1),  # "1"
    (-1, -1),  # "2"
    (3, -1),  # "0p": pass->pp, bet->pb (network)
    (3, -1),  # "1p"
    (3, -1),  # "2p"
    (4, 5),  # "0b": pass->bp, bet->bb
    (4, 5),  # "1b"
    (4, 5),  # "2b"
    (7, 8),  # "0pb": pass->pbp, bet->pbb
    (7, 8),  # "1pb"
    (7, 8),  # "2pb"
]
# Player per infoset: 0 for 0,1,2,9,10,11; 1 for 3,4,5,6,7,8
_INFOSET_PLAYER: list[int] = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
_INFOSET_CARD: list[int] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def _terminal_value(infoset_idx: int, action: int) -> float:
    """Expected return for (infoset, action) when action leads to terminal."""
    hist = _INFOSET_ACTION_TO_TERM_HIST[infoset_idx][action]
    if hist < 0:
        return 0.0  # Unused
    deal_idxs = _INFOSET_DEAL_IDXS[infoset_idx]
    player = _INFOSET_PLAYER[infoset_idx]
    total = 0.0
    for d in deal_idxs:
        total += RETURNS_ARR[d, hist, player]
    return total / len(deal_idxs)


TERMINAL_ACTION_VALUES: np.ndarray = np.zeros(
    (NUM_INFOSETS, NUM_ACTIONS), dtype=np.float32
)
for i in range(NUM_INFOSETS):
    for a in range(NUM_ACTIONS):
        if _INFOSET_ACTION_TO_TERM_HIST[i][a] >= 0:
            TERMINAL_ACTION_VALUES[i, a] = _terminal_value(i, a)


def policy_from_value_network(
    network: nn.Module,
    params: dict,
) -> FloatArray:
    """
    Greedy policy: at each infoset, take action that maximizes value (from network
    or terminal returns). Uses uniform beliefs for PBS encoding.
    """
    uniform = jnp.ones(3, dtype=jnp.float32) / 3.0
    pbs_p = PBSState(
        public_state_idx=jnp.int32(1),
        belief_p0=uniform,
        belief_p1=uniform,
    )
    pbs_b = PBSState(
        public_state_idx=jnp.int32(2),
        belief_p0=uniform,
        belief_p1=uniform,
    )
    pbs_pb = PBSState(
        public_state_idx=jnp.int32(3),
        belief_p0=uniform,
        belief_p1=uniform,
    )
    enc_p = encode_pbs(pbs_p)
    enc_b = encode_pbs(pbs_b)
    enc_pb = encode_pbs(pbs_pb)
    v_p = network.apply(params, enc_p)
    v_b = network.apply(params, enc_b)
    v_pb = network.apply(params, enc_pb)

    term_vals = jnp.array(TERMINAL_ACTION_VALUES)
    legal = _get_tree_arrays()["legal_actions_infoset"]

    # For each infoset: value of pass (action 0) and bet (action 1)
    # Infosets 0,1,2: pass->v_p, bet->v_b
    # Infosets 3,4,5: pass->term, bet->v_pb (P1: val_idx 3,4,5)
    # Infosets 6,7,8: pass->term, bet->term
    # Infosets 9,10,11: pass->term, bet->term
    v_pass = jnp.concatenate(
        [
            v_p[:3],
            term_vals[3:6, 0],
            term_vals[6:9, 0],
            term_vals[9:12, 0],
        ]
    )
    v_bet = jnp.concatenate(
        [
            v_b[:3],
            v_pb[3:6],
            term_vals[6:9, 1],
            term_vals[9:12, 1],
        ]
    )

    best = jnp.argmax(jnp.stack([v_pass, v_bet], axis=-1), axis=-1)
    policy = jnp.eye(NUM_ACTIONS, dtype=jnp.float32)[best]
    policy = jnp.where(legal, policy, 0.0)
    row_sum = jnp.sum(policy, axis=-1, keepdims=True)
    policy = jnp.where(row_sum > 1e-8, policy / row_sum, 0.5)
    return jnp.where(legal, policy, 0.0)


def policy_from_value_params(
    num_cfr_iterations: int = 64,
) -> FloatArray:
    """Run CFR-D from game root and return average policy (12, 2)."""
    _, cum_pol, _, _ = run_cfr_d(
        root_hist_idx=jnp.int32(0),
        num_iterations=num_cfr_iterations,
        linear_averaging=True,
    )
    tree = _get_tree_arrays()
    return get_average_policy_from_cfr(cum_pol, tree["legal_actions_infoset"])


def run_cfr_d(
    root_hist_idx: IntArray,
    num_iterations: int,
    linear_averaging: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Run CFR-D for num_iterations from root_hist_idx. Returns (final_regret, final_cum_policy, root_values_avg, final_values)."""
    cumulative_regret = jnp.zeros((NUM_INFOSETS, NUM_ACTIONS), dtype=jnp.float32)
    cumulative_policy = jnp.zeros((NUM_INFOSETS, NUM_ACTIONS), dtype=jnp.float32)
    root_values_sum = jnp.zeros(2, dtype=jnp.float32)

    def body(carry, t):
        cum_reg, cum_pol, rv_sum = carry
        it = t + 1
        new_reg, new_pol, rv, values = cfr_d_one_iteration(
            cum_reg, cum_pol, root_hist_idx, it, linear_averaging
        )
        rv_sum = rv_sum + rv
        return (new_reg, new_pol, rv_sum), (rv, values)

    (final_reg, final_pol, rv_sum), (_, values_history) = jax.lax.scan(
        body,
        (cumulative_regret, cumulative_policy, root_values_sum),
        jnp.arange(num_iterations),
    )
    root_values_avg = rv_sum / num_iterations
    final_values = values_history[-1]
    return final_reg, final_pol, root_values_avg, final_values


def selfplay_step(
    pbs: PBSState,
    rng: PRNGKeyArray,
    num_cfr_iterations: int,
    random_action_prob: float,
) -> tuple[FloatArray, FloatArray, PBSState, IntArray]:
    """One self-play step: solve subgame rooted at pbs, collect data, sample leaf."""
    root_hist_idx = jnp.take(
        jnp.array(PUBLIC_STATE_TO_HIST_IDX),
        jnp.int32(pbs.public_state_idx),
    )
    cum_reg, cum_pol, _, final_values = run_cfr_d(
        root_hist_idx=root_hist_idx,
        num_iterations=num_cfr_iterations,
        linear_averaging=True,
    )
    legal = _get_tree_arrays()["legal_actions_infoset"]
    policy = jax.vmap(
        lambda r, m: regret_matching(r, m),
        in_axes=(0, 0),
    )(cum_reg, legal)

    current_pbs = pbs
    pbs_enc = encode_pbs(current_pbs)

    # Use PBS-specific values as target (per ReBeL: add {βr, v(βr)} for root of subgame)
    values_target = _values_at_pbs_to_target(final_values, current_pbs.public_state_idx)

    rng, rng_act0, rng_act1, rng_act2, rng_act3, rng_act4 = jax.random.split(rng, 6)
    explore0 = jax.random.bernoulli(rng_act0, random_action_prob)

    infoset0 = jax.random.randint(rng_act1, (), 0, 3)
    probs0 = policy[infoset0]
    probs0 = jnp.where(legal[infoset0], probs0, 0.0)
    probs0 = probs0 / (jnp.sum(probs0) + 1e-8)
    action0_policy = jax.random.categorical(rng_act2, jnp.log(probs0 + 1e-8))
    action0_random = jax.random.randint(rng_act3, (), 0, NUM_ACTIONS)
    action0 = jnp.where(explore0, action0_random, action0_policy)

    next_pbs = update_belief_after_action(current_pbs, action0, policy)

    def from_p_sample():
        rng_a, rng_b = jax.random.split(rng_act4)
        explore1 = jax.random.bernoulli(rng_a, random_action_prob)
        infoset1 = 3 + jax.random.randint(rng_b, (), 0, 3)
        probs1 = policy[infoset1]
        probs1 = jnp.where(legal[infoset1], probs1, 0.0)
        probs1 = probs1 / (jnp.sum(probs1) + 1e-8)
        rng_c, rng_d = jax.random.split(rng_b)
        act_pol = jax.random.categorical(rng_c, jnp.log(probs1 + 1e-8))
        act_rand = jax.random.randint(rng_d, (), 0, NUM_ACTIONS)
        act = jnp.where(explore1, act_rand, act_pol)
        return update_belief_after_action(next_pbs, act, policy)

    next_pbs = jax.lax.cond(
        next_pbs.public_state_idx == 1,
        from_p_sample,
        lambda: next_pbs,
    )
    is_terminal_val = next_pbs.public_state_idx == 2
    return pbs_enc, values_target, next_pbs, is_terminal_val.astype(jnp.int32)


# =============================================================================
# Training loop
# =============================================================================


class ReplayBufferState(NamedTuple):
    pbs: FloatArray
    values: FloatArray
    seen: IntArray
    size: IntArray


class RunnerState(NamedTuple):
    value_train_state: TrainState
    replay_buffer: ReplayBufferState
    pbs_root: FloatArray
    update_step: IntArray
    rng: PRNGKeyArray


def huber_loss(x: FloatArray, delta: float = 1.0) -> FloatArray:
    """Huber loss for value network (per ReBeL paper)."""
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x * x,
        delta * (abs_x - 0.5 * delta),
    )


def make_train(config: dict) -> Callable[[PRNGKeyArray, IntArray], RunnerState]:
    capacity = config["replay_capacity"]
    batch_size = config["batch_size"]
    num_subgames_per_update = config["num_subgames_per_update"]
    num_value_train_steps = config["num_value_train_steps"]
    cfr_iterations = config["cfr_iterations"]
    random_action_prob = config["random_action_prob"]

    def init_replay_buffer() -> ReplayBufferState:
        return ReplayBufferState(
            pbs=jnp.zeros((capacity, PBS_INPUT_DIM), dtype=jnp.float32),
            values=jnp.zeros((capacity, VALUE_OUTPUT_DIM), dtype=jnp.float32),
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
        not_full = k < capacity
        write_idx = jnp.where(not_full, k, j)
        should_write = not_full | (j < capacity)
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
                size=jnp.minimum(capacity, buffer.size + 1),
            ),
            rng,
        )

    def sample_batch(
        rng: PRNGKeyArray,
        buffer: ReplayBufferState,
        batch_size: int,
    ) -> tuple[FloatArray, FloatArray]:
        max_size = jnp.maximum(buffer.size, 1)
        indices = jax.random.randint(rng, (batch_size,), 0, max_size, dtype=jnp.int32)
        return buffer.pbs[indices], buffer.values[indices]

    log_interval = config.get("log_interval", 100)

    def train(rng: PRNGKeyArray, seed: IntArray) -> RunnerState:
        rng, rng_init = jax.random.split(rng)
        network = ValueNetworkMLP(hidden_dim=config["value_hidden_dim"])
        dummy_pbs = encode_pbs(root_pbs())
        params = network.init(rng_init, dummy_pbs)

        tx = optax.adam(learning_rate=config["value_lr"])
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        replay_buffer = init_replay_buffer()
        pbs_root_enc = encode_pbs(root_pbs())

        def value_loss(params, pbs_batch, values_batch):
            pred = network.apply(params, pbs_batch)
            err = pred - values_batch
            return jnp.mean(huber_loss(err))

        def train_step(
            train_state: TrainState,
            pbs_batch: FloatArray,
            values_batch: FloatArray,
        ) -> tuple[TrainState, FloatArray]:
            loss_val, grads = jax.value_and_grad(value_loss)(
                train_state.params, pbs_batch, values_batch
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss_val

        def update_step(runner_state: RunnerState, _) -> tuple[RunnerState, dict]:
            train_state = runner_state.value_train_state
            buffer = runner_state.replay_buffer
            pbs_enc = runner_state.pbs_root
            rng = runner_state.rng

            def run_selfplay(carry, _):
                ts, buf, pb_enc, r = carry
                pbs_struct = decode_pbs(pb_enc)
                r, r_sp = jax.random.split(r)
                pbs_out, values_target, next_pbs, is_term = selfplay_step(
                    pbs_struct,
                    r_sp,
                    cfr_iterations,
                    random_action_prob,
                )
                next_enc = encode_pbs(next_pbs)
                r, r_ap = jax.random.split(r)
                buf, _ = append_one(buf, pbs_out, values_target, r_ap)
                next_root = jnp.where(is_term.astype(jnp.bool_), pbs_root_enc, next_enc)
                return (ts, buf, next_root, r), ()

            (train_state, buffer, pbs_enc, rng), _ = lax.scan(
                run_selfplay,
                (train_state, buffer, pbs_enc, rng),
                None,
                num_subgames_per_update,
            )

            def train_value_epoch(carry, _):
                ts, r = carry
                r, r_batch = jax.random.split(r)
                pbs_b, values_b = sample_batch(r_batch, buffer, batch_size)
                ts, loss = train_step(ts, pbs_b, values_b)
                return (ts, r), loss

            rng, rng_train = jax.random.split(rng)
            (train_state, _), losses = lax.scan(
                train_value_epoch,
                (train_state, rng_train),
                None,
                num_value_train_steps,
            )
            mean_loss = jnp.mean(losses)

            metric = {
                "value_loss": mean_loss,
                "replay_size": buffer.size.astype(jnp.float32),
                "update_step": runner_state.update_step,
                "total_samples": buffer.seen.astype(jnp.float32),
            }

            should_log = (runner_state.update_step % log_interval) == 0
            policy_arr = lax.cond(
                should_log,
                lambda: policy_from_value_network(network, train_state.params),
                lambda: jnp.zeros((12, 2), dtype=jnp.float32),
            )

            def expl_callback(sid: int, m: dict, policy: np.ndarray, st: int) -> None:
                m = dict(m)
                if int(st) % log_interval == 0:
                    # When vmapped, policy has shape (num_seeds, 12, 2); extract (12, 2)
                    policy_2d = policy[0] if policy.ndim == 3 else policy

                    def policy_fn(key):
                        idx = list(INFOSET_KEYS).index(key)
                        return policy_2d[idx].astype(np.float64)

                    m["exploitability"] = exploitability(policy_fn)
                np_m = {k: np.array(v) for k, v in m.items()}
                LOGGER.log(int(sid), np_m)

            jax.experimental.io_callback(
                expl_callback,
                None,
                seed,
                metric,
                policy_arr,
                runner_state.update_step,
            )

            return (
                RunnerState(
                    value_train_state=train_state,
                    replay_buffer=buffer,
                    pbs_root=pbs_enc,
                    update_step=runner_state.update_step + 1,
                    rng=rng,
                ),
                metric,
            )

        initial_state = RunnerState(
            value_train_state=train_state,
            replay_buffer=replay_buffer,
            pbs_root=pbs_root_enc,
            update_step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )

        final_state, _ = lax.scan(
            update_step,
            initial_state,
            None,
            config["num_update_steps"],
        )
        return final_state

    return train


LOGGER: WandbMultiLogger | None = None


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
        _ = jax.block_until_ready(train_vjit(rng_seeds, exp_ids))
    finally:
        LOGGER.finish()
        print("Finished.")


if __name__ == "__main__":
    main()
