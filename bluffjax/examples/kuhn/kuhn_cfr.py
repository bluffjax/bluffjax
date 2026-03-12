"""
Counterfactual Regret Minimization (CFR) for Kuhn Poker.

CFR learns a Nash equilibrium by iteratively:
1. Computing the current strategy via regret matching on cumulative regrets
2. Traversing the game tree to compute counterfactual values
3. Updating cumulative regrets and cumulative policy

The average policy converges to a Nash equilibrium. Exploitability is logged
each iteration using the OpenSpiel-compatible exploitability metric.
"""

import itertools
from collections import defaultdict
from typing import Callable

import numpy as np

from bluffjax.utils.game_utils.kuhn_exploitability import (
    INFOSETS,
    ACTION_PASS,
    ACTION_BET,
    infoset_key,
    get_returns,
    is_terminal,
    get_legal_actions,
    get_current_player,
    exploitability,
)


def _regret_matching(
    cumulative_regret: dict[int, float],
    legal_actions: list[int],
) -> np.ndarray:
    """Regret matching: strategy proportional to positive regrets, else uniform."""
    sum_pos = sum(max(0.0, cumulative_regret.get(a, 0.0)) for a in legal_actions)
    probs = np.zeros(2, dtype=np.float64)
    if sum_pos > 0:
        for a in legal_actions:
            probs[a] = max(0.0, cumulative_regret.get(a, 0.0)) / sum_pos
    else:
        for a in legal_actions:
            probs[a] = 1.0 / len(legal_actions)
    return probs


def _cfr_traversal(
    hands: tuple[int, int],
    history: tuple[int, ...],
    reach_probs: np.ndarray,
    infoset_nodes: dict[str, dict],
    current_policy: dict[str, np.ndarray],
    iteration: int,
    linear_averaging: bool,
    update_player: int | None,
) -> np.ndarray:
    """
    Traverse game tree and update cumulative regrets and policy.
    reach_probs: [reach_p0, reach_p1] (probability each player reached this node)
    update_player: If set, only update regrets for this player (alternating). If None, update both.
    Returns: value for each player at this node.
    """
    if is_terminal(history):
        ret = get_returns(hands, history)
        return np.array([ret[0], ret[1]], dtype=np.float64)

    current = get_current_player(history)
    card = hands[current]
    infok = infoset_key(card, history)
    legal = get_legal_actions(history)

    if infok not in infoset_nodes:
        infoset_nodes[infok] = {
            "cumulative_regret": defaultdict(float),
            "cumulative_policy": defaultdict(float),
        }
    node = infoset_nodes[infok]
    policy_probs = current_policy[infok]

    state_value = np.zeros(2)
    child_values = {}

    for a in legal:
        new_reach = reach_probs.copy()
        new_reach[current] *= policy_probs[a]
        child_val = _cfr_traversal(
            hands,
            history + (a,),
            new_reach,
            infoset_nodes,
            current_policy,
            iteration,
            linear_averaging,
            update_player,
        )
        state_value += policy_probs[a] * child_val
        child_values[a] = child_val

    # Only update if we're updating this player (alternating) or updating all (simultaneous)
    should_update = update_player is None or current == update_player

    if should_update:
        cf_reach = 1.0
        for p in range(2):
            if p != current:
                cf_reach *= reach_probs[p]

        for a in legal:
            regret = cf_reach * (child_values[a][current] - state_value[current])
            node["cumulative_regret"][a] += regret
            if linear_averaging:
                node["cumulative_policy"][a] += (
                    iteration * reach_probs[current] * policy_probs[a]
                )
            else:
                node["cumulative_policy"][a] += reach_probs[current] * policy_probs[a]

    return state_value


def _run_cfr_iteration(
    infoset_nodes: dict[str, dict],
    current_policy: dict[str, np.ndarray],
    iteration: int,
    linear_averaging: bool,
    alternating_updates: bool,
) -> float:
    """One CFR iteration: update policy from regrets, then traverse (alternating or simultaneous)."""
    total_value = 0.0

    if alternating_updates:
        for player in range(2):
            for infok in INFOSETS:
                if infok in infoset_nodes:
                    node = infoset_nodes[infok]
                    legal = [ACTION_PASS, ACTION_BET]
                    current_policy[infok] = _regret_matching(
                        dict(node["cumulative_regret"]),
                        legal,
                    )
            for hands in itertools.permutations([0, 1, 2], 2):
                reach = np.ones(2, dtype=np.float64)
                val = _cfr_traversal(
                    hands,
                    (),
                    reach,
                    infoset_nodes,
                    current_policy,
                    iteration,
                    linear_averaging,
                    player,
                )
                total_value += val[0]
    else:
        for infok in INFOSETS:
            if infok in infoset_nodes:
                node = infoset_nodes[infok]
                legal = [ACTION_PASS, ACTION_BET]
                current_policy[infok] = _regret_matching(
                    dict(node["cumulative_regret"]),
                    legal,
                )
        for hands in itertools.permutations([0, 1, 2], 2):
            reach = np.ones(2, dtype=np.float64)
            val = _cfr_traversal(
                hands,
                (),
                reach,
                infoset_nodes,
                current_policy,
                iteration,
                linear_averaging,
                None,
            )
            total_value += val[0]

    return total_value / 6


def _average_policy_from_nodes(
    infoset_nodes: dict[str, dict],
) -> Callable[[str], np.ndarray]:
    """Build average policy from cumulative policy in infoset nodes."""

    def policy(infok: str) -> np.ndarray:
        if infok not in infoset_nodes:
            return np.array([0.5, 0.5], dtype=np.float64)
        node = infoset_nodes[infok]
        cum = node["cumulative_policy"]
        total = sum(cum.values())
        if total <= 0:
            return np.array([0.5, 0.5], dtype=np.float64)
        probs = np.zeros(2, dtype=np.float64)
        for a in [ACTION_PASS, ACTION_BET]:
            probs[a] = cum.get(a, 0.0) / total
        return probs

    return policy


def run_cfr(
    num_iterations: int = 10_000,
    linear_averaging: bool = False,
    alternating_updates: bool = True,
    log_interval: int = 100,
) -> tuple[Callable[[str], np.ndarray], list[tuple[int, float]]]:
    """
    Run CFR on Kuhn poker.

    Args:
        num_iterations: Number of CFR iterations.
        linear_averaging: If True, use linear (iteration-weighted) averaging.
        alternating_updates: If True, alternate updates between players (faster convergence).
        log_interval: Log exploitability every this many iterations.

    Returns:
        average_policy: The converged average policy.
        exploitability_log: List of (iteration, exploitability) tuples.
    """
    infoset_nodes: dict[str, dict] = {}
    current_policy: dict[str, np.ndarray] = {}

    # Initialize with uniform
    for infok in INFOSETS:
        current_policy[infok] = np.array([0.5, 0.5], dtype=np.float64)
        infoset_nodes[infok] = {
            "cumulative_regret": defaultdict(float),
            "cumulative_policy": defaultdict(float),
        }

    exploitability_log: list[tuple[int, float]] = []

    # Log initial (uniform) exploitability
    avg_policy_0 = _average_policy_from_nodes(infoset_nodes)
    expl_0 = exploitability(avg_policy_0)
    exploitability_log.append((0, expl_0))
    print(f"Iteration      0: exploitability = {expl_0:.6f} (uniform)")

    for it in range(1, num_iterations + 1):
        _run_cfr_iteration(
            infoset_nodes,
            current_policy,
            it,
            linear_averaging,
            alternating_updates,
        )

        if it % log_interval == 0 or it == num_iterations:
            avg_policy = _average_policy_from_nodes(infoset_nodes)
            expl = exploitability(avg_policy)
            exploitability_log.append((it, expl))
            print(f"Iteration {it:6d}: exploitability = {expl:.6f}")

    return _average_policy_from_nodes(infoset_nodes), exploitability_log


def main() -> None:
    num_iterations = 100
    log_interval = 10

    print("CFR for Kuhn Poker")
    print("=" * 50)
    avg_policy, expl_log = run_cfr(
        num_iterations=num_iterations,
        linear_averaging=False,
        log_interval=log_interval,
    )

    final_expl = float(expl_log[-1][1])
    print("=" * 50)
    print(f"Final exploitability: {final_expl:.6f} (expected ~0 at Nash)")


if __name__ == "__main__":
    main()
