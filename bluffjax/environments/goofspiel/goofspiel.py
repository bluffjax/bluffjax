"""
Goofspiel (Game of Pure Strategy) environment.

A simultaneous-move bidding card game where each player has cards 1-13,
a random prize deck is revealed one card per round, and players simultaneously
bid. Highest bid wins the prize (value 1-13); ties discard the prize.
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


@struct.dataclass
class GoofspielState:
    """Goofspiel game state."""

    player_hands: BoolArray  # (num_agents, 13) - True if card still in hand
    deck: IntArray  # (13,) - shuffled 0..12, order of revelation
    current_round: IntArray  # 0..12, which round we're in
    points: FloatArray  # (num_agents,) - cumulative points
    winners: IntArray  # (num_agents,) - which player won the round
    absorbing: BoolArray  # (num_agents,)
    done: bool
    timestep: int


class Goofspiel(ParallelEnv):
    """Goofspiel parallel environment."""

    def __init__(self, num_agents: int = 2, horizon: int = 14, num_decks=1) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 13
        self.num_cards = num_decks * 13
        self.obs_dim = self.deck_size * 3
        self.num_actions = self.deck_size

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: GoofspielState) -> FloatArray:
        """
        Observation per agent: (num_agents, obs_dim).
        - Current prize card: one-hot (13)
        - Self cards used: binary (13)
        - Opponent cards used: binary (13)
        """
        # Current prize card one-hot (only valid when not done)
        current_prize = state.deck[state.current_round]
        prize_onehot = jax.nn.one_hot(current_prize, self.deck_size, dtype=jnp.float32)

        # Cards used by self = complement of hand
        cards_played = (~state.player_hands).astype(jnp.float32)

        # Cards used by opponent: for agent i, opponent is (i+1) % num_agents
        cards_played_opp = jnp.roll(cards_played, 1, axis=0)

        obs = jnp.concatenate(
            [
                jnp.broadcast_to(prize_onehot, (self.num_agents, self.deck_size)),
                cards_played,
                cards_played_opp,
            ],
            axis=1,
        )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: GoofspielState) -> BoolArray:
        """Available actions: (num_agents, 13) - True if card is in hand."""
        return jnp.where(
            state.done,
            jnp.zeros((self.num_agents, self.num_actions), dtype=bool),
            state.player_hands,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[GoofspielState, FloatArray]:
        """Reset the environment."""
        deck_shuffled = jax.random.permutation(rng, self.deck_size)
        player_hands = jnp.ones((self.num_agents, self.deck_size), dtype=bool)
        points = jnp.zeros(self.num_agents, dtype=jnp.float32)

        state = GoofspielState(
            player_hands=player_hands,
            deck=deck_shuffled,
            current_round=jnp.int32(0),
            points=points,
            winners=jnp.zeros(self.num_agents, dtype=bool),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=False,
            timestep=0,
        )
        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: GoofspielState,
        action: IntArray,
    ) -> tuple[
        GoofspielState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Execute one step: process simultaneous bids, update state."""

        # Prize value for this round (1-13)
        prize_value = (state.deck[state.current_round] + 1).astype(jnp.float32)

        # Winner: highest bid, no ties
        max_bid = jnp.max(action)
        num_max_bidders = jnp.sum(action == max_bid)
        round_winner_mask = (action == max_bid) & (num_max_bidders == 1)

        rewards = jnp.where(round_winner_mask, prize_value, 0.0)

        # Update points
        new_points = state.points + rewards

        # Remove played cards from hands
        new_hands = state.player_hands.at[jnp.arange(self.num_agents), action].set(
            False
        )

        # Advance round
        new_round = state.current_round + 1
        game_done = new_round >= self.num_cards
        new_absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        new_winners = jnp.where(
            game_done, new_points == jnp.max(new_points), state.winners
        )

        next_state = GoofspielState(
            player_hands=new_hands,
            deck=state.deck,
            current_round=new_round,
            points=new_points,
            winners=new_winners,
            absorbing=new_absorbing,
            done=game_done,
            timestep=state.timestep + 1,
        )
        next_obs = self.obs_from_state(next_state)
        return next_state, next_obs, rewards, new_absorbing, game_done, {}

    def observation_space(self) -> Discrete:
        """Observation space: (num_agents, 39) float array."""
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        """Action space: 13 discrete choices per agent."""
        return Discrete(self.num_actions)
