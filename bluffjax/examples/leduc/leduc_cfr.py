"""CFR on Leduc Poker.

This runs vanilla tabular CFR (full-tree traversals) and logs exploitability
of the learned average policy during training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from bluffjax.utils.game_utils import leduc_exploitability as leduc


class CFRSolver:
    def __init__(self) -> None:
        # regrets[player][infoset_key][action] -> cumulative regret
        self.regrets = [defaultdict(lambda: defaultdict(float)) for _ in range(2)]
        # strategy_sum[player][infoset_key][action] -> cumulative average-strategy mass
        self.strategy_sum = [defaultdict(lambda: defaultdict(float)) for _ in range(2)]

    @staticmethod
    def _uniform_over(actions: tuple[int, ...]) -> dict[int, float]:
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    def _regret_matching(
        self, player: int, infoset_key: tuple, legal: tuple[int, ...]
    ) -> dict[int, float]:
        regrets = self.regrets[player][infoset_key]
        positive = np.array([max(regrets[a], 0.0) for a in legal], dtype=np.float64)
        if positive.sum() <= 0:
            return self._uniform_over(legal)
        probs = positive / positive.sum()
        return {a: float(probs[i]) for i, a in enumerate(legal)}

    def _cfr(
        self, state: leduc.LeducState, update_player: int, reach: tuple[float, float]
    ) -> float:
        if state.is_terminal():
            return state.returns()[update_player]

        if state.is_chance_node():
            v = 0.0
            for action, prob in state.chance_outcomes():
                v += prob * self._cfr(state.child(action), update_player, reach)
            return v

        current = state.current_player
        legal = state.legal_actions()
        infoset_key = state.info_state_key(current)
        strategy = self._regret_matching(current, infoset_key, legal)

        if current == update_player:
            action_values: dict[int, float] = {}
            node_value = 0.0
            for action in legal:
                next_reach = list(reach)
                next_reach[current] *= strategy[action]
                val = self._cfr(
                    state.child(action), update_player, (next_reach[0], next_reach[1])
                )
                action_values[action] = val
                node_value += strategy[action] * val

            opp = 1 - current
            for action in legal:
                regret = action_values[action] - node_value
                self.regrets[current][infoset_key][action] += reach[opp] * regret
            return node_value

        node_value = 0.0
        for action in legal:
            next_reach = list(reach)
            next_reach[current] *= strategy[action]
            node_value += strategy[action] * self._cfr(
                state.child(action), update_player, (next_reach[0], next_reach[1])
            )
        return node_value

    def _accumulate_average_policy(
        self, state: leduc.LeducState, reach: tuple[float, float]
    ) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                self._accumulate_average_policy(
                    state.child(action), (reach[0], reach[1])
                )
            return

        player = state.current_player
        legal = state.legal_actions()
        infoset_key = state.info_state_key(player)
        strategy = self._regret_matching(player, infoset_key, legal)

        # Standard average-strategy accumulation: weight by player's realization.
        for action in legal:
            self.strategy_sum[player][infoset_key][action] += (
                reach[player] * strategy[action]
            )

        for action in legal:
            next_reach = list(reach)
            next_reach[player] *= strategy[action]
            self._accumulate_average_policy(
                state.child(action), (next_reach[0], next_reach[1])
            )

    def run_iteration(self) -> None:
        root = leduc.initial_state()
        for p in (0, 1):
            self._cfr(root, p, (1.0, 1.0))
        self._accumulate_average_policy(root, (1.0, 1.0))

    def average_policy(self, state: leduc.LeducState) -> dict[int, float]:
        legal = state.legal_actions()
        if not legal:
            return {}
        if state.is_chance_node():
            return {a: p for a, p in state.chance_outcomes()}

        player = state.current_player
        infoset_key = state.info_state_key(player)
        sums = self.strategy_sum[player][infoset_key]
        total = sum(sums[a] for a in legal)
        if total <= 0:
            return self._uniform_over(legal)
        return {a: sums[a] / total for a in legal}

    def nash_conv(self) -> float:
        root = leduc.initial_state()
        on_policy = leduc._state_values(root, self.average_policy)
        br0 = leduc.BestResponseSolver(root, 0, self.average_policy).value(root)
        br1 = leduc.BestResponseSolver(root, 1, self.average_policy).value(root)
        return (br0 - on_policy[0]) + (br1 - on_policy[1])

    def exploitability(self) -> float:
        return self.nash_conv() / 2.0


def run_cfr(iterations: int, log_every: int) -> CFRSolver:
    solver = CFRSolver()
    for it in range(1, iterations + 1):
        solver.run_iteration()
        if it == 1 or it % log_every == 0 or it == iterations:
            expl = solver.exploitability()
            print(f"iter={it:6d} exploitability={expl:.6f}")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="CFR on Leduc Poker.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    solver = run_cfr(args.iterations, args.log_every)
    final_expl = solver.exploitability()
    print(
        f"\nfinal exploitability after {args.iterations} iterations: {final_expl:.6f}"
    )


if __name__ == "__main__":
    main()
