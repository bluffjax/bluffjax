"""
Kuhn Poker environment.
This code models the dynamics of Kuhn Poker[1]:
https://en.wikipedia.org/wiki/Kuhn_poker

State:
0,1,2 = Jack, Queen, King

Actions:
0 = check
1 = raise

[1] Kuhn, Harold W. "A simplified two-person poker." Contributions to the Theory of Games 1 (2016): 97-103.
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


@struct.dataclass
class KuhnState:
    """
    Kuhn Poker state.
    """

    agent_hands: IntArray
    pot: IntArray
    start_player_idx: IntArray
    current_player_idx: IntArray
    returns: FloatArray
    absorbing: BoolArray
    all_absorbing: bool
    done: bool
    timestep: IntArray


class KuhnPoker(AECEnv):
    def __init__(self) -> None:
        super().__init__(num_agents=2, horizon=10)
        self.ranks = jnp.array([0, 1, 2], dtype=jnp.int32)
        self.num_ranks = 3
        self.base_pot = 2
        self.obs_dim = 9  # 3 hand + 2 self_chips + 2 opp_chips + 2 player_idx
        self.num_actions = 2  # check, raise

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> KuhnState:
        """Return the first state of the game"""
        rng_shuffle, rng_player = jax.random.split(rng)
        start_player_idx = jax.random.randint(
            rng_player, shape=(), minval=0, maxval=self.num_agents, dtype=jnp.int32
        )
        hands = jax.random.permutation(rng_shuffle, self.ranks)[: self.num_agents]
        state = KuhnState(
            agent_hands=hands,
            pot=jnp.ones(self.num_agents, dtype=jnp.int32),
            start_player_idx=start_player_idx,
            current_player_idx=start_player_idx,
            returns=jnp.zeros(self.num_agents, dtype=jnp.float32),
            absorbing=jnp.zeros(self.num_agents, dtype=jnp.bool),
            all_absorbing=False,
            done=False,
            timestep=jnp.int32(0),
        )
        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state) -> FloatArray:
        """
        Observation format (9 dims):
        agent_hand (3)
        + self_chips_oh (2)
        + opp_chips_oh (2)
        + player_relative_idx_oh (2) (first to act or second to act)
        """
        agent_hand = jax.nn.one_hot(
            state.agent_hands[state.current_player_idx],
            self.num_ranks,
            dtype=jnp.float32,
        )

        # self chips (1 or 2) -> index 0 or 1
        self_chips_oh = jax.nn.one_hot(
            state.pot[state.current_player_idx] - 1,
            2,
            dtype=jnp.float32,
        )
        # opponent chips (1 or 2) -> index 0 or 1
        opp_chips_oh = jax.nn.one_hot(
            state.pot[1 - state.current_player_idx] - 1,
            2,
            dtype=jnp.float32,
        )

        player_relative_idx = (
            state.current_player_idx - state.start_player_idx
        ) % self.num_agents
        player_relative_idx_oh = jax.nn.one_hot(
            player_relative_idx, self.num_agents, dtype=jnp.float32
        )
        obs = jnp.concatenate(
            [
                agent_hand,
                self_chips_oh,
                opp_chips_oh,
                player_relative_idx_oh,
            ],
            axis=0,
        )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: KuhnState) -> BoolArray:
        return jnp.where(
            state.done, jnp.zeros(2, dtype=jnp.bool), jnp.ones(2, dtype=jnp.bool)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: KuhnState, action: IntArray
    ) -> tuple[KuhnState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]]:
        bet_played = action > 0
        higher_idx = jnp.argmax(state.agent_hands).astype(jnp.int32)

        is_second_action = state.timestep == 1
        is_third_action = state.timestep == 2

        first_was_check = state.pot.sum() == 2
        first_was_bet = state.pot.sum() == 3

        showdown = (
            (is_second_action & first_was_check & ~bet_played)
            | (is_second_action & first_was_bet & bet_played)
            | (is_third_action & bet_played)
        )
        bettor_wins = (is_second_action & first_was_bet & ~bet_played) | (
            is_third_action & ~bet_played
        )

        next_pot = state.pot.at[state.current_player_idx].add(action)
        next_player_idx = (state.current_player_idx + 1) % self.num_agents

        game_winner = jnp.where(
            showdown,
            self.agent_idxs == higher_idx,
            jnp.where(
                bettor_wins,
                self.agent_idxs == next_player_idx,
                jnp.zeros(self.num_agents, dtype=jnp.bool),
            ),
        )

        pot_total = next_pot.sum().astype(jnp.float32)
        profits = jnp.where(
            game_winner,
            pot_total - next_pot.astype(jnp.float32),
            -next_pot.astype(jnp.float32),
        )
        rewards = jnp.where(
            game_winner.any(), profits, jnp.zeros(self.num_agents, dtype=jnp.float32)
        )
        returns = state.returns + rewards
        next_timestep = state.timestep + 1
        next_absorbing = jnp.tile(game_winner.any(), (self.num_agents))
        next_all_absorbing = next_absorbing.all()
        next_done = next_all_absorbing | (next_timestep > self.horizon)

        next_state = KuhnState(
            agent_hands=state.agent_hands,
            pot=next_pot,
            start_player_idx=state.start_player_idx,
            current_player_idx=next_player_idx,
            returns=returns,
            absorbing=next_absorbing,
            all_absorbing=next_all_absorbing,
            done=next_done,
            timestep=next_timestep,
        )
        next_obs = self.obs_from_state(next_state)
        info = {"timestep": next_timestep, "returns": returns}
        return (next_state, next_obs, rewards, next_absorbing, next_done, info)
