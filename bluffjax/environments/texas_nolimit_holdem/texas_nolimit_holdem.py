"""
Texas No-Limit Hold'em environment.
"""

from jax._src.basearray import Array
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
from bluffjax.utils.game_utils.poker_utils import _compare_hands


@struct.dataclass
class TexasNoLimitHoldEmState:
    flop_cards: FloatArray
    turn_card: FloatArray
    river_card: FloatArray
    agent_cards: FloatArray
    chips_in: FloatArray
    round_raised: FloatArray
    remaining_chips: FloatArray
    small_blind_idx: IntArray
    current_player_idx: IntArray
    stage: IntArray  # 0=preflop, 1=flop, 2=turn, 3=river
    folded: BoolArray  # Whether each player has folded
    all_in: BoolArray  # Whether each player is all-in
    not_raise_num: IntArray  # Number of players who have checked/called without raising
    absorbing: BoolArray
    done: bool
    timestep: int


class TexasNoLimitHoldem(AECEnv):
    def __init__(
        self, num_agents: int = 2, horizon: int = 100_000, init_chips: int = 100
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 52
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind
        self.init_chips = init_chips
        self.obs_dim = self.deck_size + 2
        self.num_actions = 5  # check/call, raise half, raise pot, all in, fold

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: TexasNoLimitHoldEmState) -> FloatArray:
        """
        Extract observation for a specific agent in rlcard no-limit holdem style.

        Observation format (54 dimensions):
        - Dimensions 0-51: One-hot encoding of cards (hand cards + public cards)
        - Dimension 52: Current player's chips in the pot
        - Dimension 53: Max chips in the pot among all players

        Args:
            state: Current game state

        Returns:
            observation: 54-dimensional observation vector
        """
        # Get public cards based on current stage using lax.switch
        # Pre-flop (stage 0): no public cards
        # Flop (stage 1): 3 cards
        # Turn (stage 2): 4 cards
        # River (stage 3): 5 cards
        stage = state.stage

        # Extract cards from state for use in switch branches
        hand_cards = state.agent_cards[state.current_player_idx]
        flop = state.flop_cards
        turn = state.turn_card
        river = state.river_card

        # Initialize observation vector (54 dimensions)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        obs = obs.at[hand_cards].set(1.0)

        obs = jnp.where(stage >= 1, obs.at[flop].set(1.0), obs)
        obs = jnp.where(stage >= 2, obs.at[turn].set(1.0), obs)
        obs = jnp.where(stage >= 3, obs.at[river].set(1.0), obs)

        current_player = state.current_player_idx
        obs = obs.at[52].set(state.chips_in[current_player].astype(jnp.float32))
        obs = obs.at[53].set(jnp.max(state.chips_in).astype(jnp.float32))

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: TexasNoLimitHoldEmState) -> BoolArray:
        """
        Get available actions mask for the current player.

        Actions (in order):
        0: check/call
        1: raise half pot
        2: raise pot
        3: all in
        4: fold

        Rules:
        - Check/call is always allowed for active players
        - Raises depend on remaining chips and current round bet
        - Fold is always allowed for active players

        Args:
            state: Current game state

        Returns:
            avail_actions: Boolean mask of shape (5,) indicating which actions are legal
        """
        current_player = state.current_player_idx
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)

        is_active = (
            ~state.folded[current_player]
            & ~state.all_in[current_player]
            & (player_remain > 0)
        )

        # Initialize all actions as False, then enable legal ones
        avail_actions = jnp.zeros(5, dtype=bool)

        diff = max_round - player_round
        can_raise_any = diff < player_remain
        can_raise_pot = can_raise_any & (pot <= player_remain)
        can_raise_half = (
            can_raise_any
            & (half_pot <= player_remain)
            & ((half_pot + player_round) > max_round)
        )

        avail_actions = avail_actions.at[0].set(is_active)
        avail_actions = avail_actions.at[1].set(is_active & can_raise_half)
        avail_actions = avail_actions.at[2].set(is_active & can_raise_pot)
        avail_actions = avail_actions.at[3].set(is_active & can_raise_any)
        avail_actions = avail_actions.at[4].set(is_active)
        return avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> TexasNoLimitHoldEmState:
        """Return the first state of the game"""
        rng_shuffle, rng_player = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)
        agent_cards = shuffled_deck[: 2 * self.num_agents].reshape(-1, 2)
        flop = shuffled_deck[(2 * self.num_agents) : (2 * self.num_agents) + 3]
        turn = shuffled_deck[(2 * self.num_agents) + 3]
        river = shuffled_deck[(2 * self.num_agents) + 4]
        small_blind_agent = jax.random.randint(
            rng_player, shape=(), minval=0, maxval=self.num_agents
        )
        big_blind_agent = (small_blind_agent + 1) % self.num_agents
        chips_in = jnp.zeros(self.num_agents)
        chips_in = chips_in.at[small_blind_agent].set(self.small_blind)
        chips_in = chips_in.at[big_blind_agent].set(self.big_blind)
        round_raised = chips_in
        remaining_chips = jnp.full(self.num_agents, self.init_chips) - chips_in
        current_player_idx = (small_blind_agent + 2) % self.num_agents

        state = TexasNoLimitHoldEmState(
            flop_cards=flop,
            turn_card=turn,
            river_card=river,
            agent_cards=agent_cards,
            chips_in=chips_in,
            round_raised=round_raised,
            remaining_chips=remaining_chips,
            small_blind_idx=small_blind_agent,
            current_player_idx=current_player_idx,
            stage=jnp.int32(0),
            folded=jnp.zeros(self.num_agents, dtype=bool),
            all_in=jnp.zeros(self.num_agents, dtype=bool),
            not_raise_num=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.array(False),
            timestep=0,
        )

        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: TexasNoLimitHoldEmState, action: IntArray
    ) -> tuple[
        TexasNoLimitHoldEmState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step in the game.

        Args:
            state: Current game state
            action: Action taken by current player
                0=check/call, 1=raise_half_pot, 2=raise_pot, 3=all_in, 4=fold

        Returns:
            next_state: Updated game state
            obs: Observation for next player
            reward: Rewards for all players (zeros during game, payoffs at end)
            absorbing: Whether absorbing state reached
            done: Whether game is done
        """
        current_player = state.current_player_idx
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)
        diff = max_round - player_round

        # Process action using switch
        def apply_bet(
            bet_amount: FloatArray,
        ) -> tuple[FloatArray, FloatArray, FloatArray, BoolArray, BoolArray]:
            new_chips_in = state.chips_in.at[current_player].add(bet_amount)
            new_round_raised = state.round_raised.at[current_player].add(bet_amount)
            new_remaining = state.remaining_chips.at[current_player].add(-bet_amount)
            new_all_in = state.all_in.at[current_player].set(
                new_remaining[current_player] <= 0
            )
            new_folded = state.folded
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
            )

        def process_check_call() -> tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            BoolArray,
            BoolArray,
            IntArray,
        ]:
            (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
            ) = apply_bet(diff)
            player_all_in = new_all_in[current_player]
            new_not_raise_num = jnp.where(
                player_all_in,
                state.not_raise_num,
                state.not_raise_num + 1,
            )
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
                new_not_raise_num,
            )

        def process_raise_half_pot() -> tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            BoolArray,
            BoolArray,
            IntArray,
        ]:
            (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
            ) = apply_bet(half_pot)
            player_all_in = new_all_in[current_player]
            new_not_raise_num = jnp.where(
                player_all_in,
                state.not_raise_num,
                jnp.int32(1),
            )
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
                new_not_raise_num,
            )

        def process_raise_pot() -> tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            BoolArray,
            BoolArray,
            IntArray,
        ]:
            (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
            ) = apply_bet(pot)
            player_all_in = new_all_in[current_player]
            new_not_raise_num = jnp.where(
                player_all_in,
                state.not_raise_num,
                jnp.int32(1),
            )
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
                new_not_raise_num,
            )

        def process_all_in() -> tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            BoolArray,
            BoolArray,
            IntArray,
        ]:
            (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
            ) = apply_bet(player_remain)
            player_all_in = new_all_in[current_player]
            new_not_raise_num = jnp.where(
                player_all_in,
                state.not_raise_num,
                jnp.int32(1),
            )
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
                new_not_raise_num,
            )

        def process_fold() -> tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            BoolArray,
            BoolArray,
            IntArray,
        ]:
            new_chips_in = state.chips_in
            new_round_raised = state.round_raised
            new_remaining = state.remaining_chips
            new_not_raise_num = state.not_raise_num
            new_folded = state.folded.at[current_player].set(True)
            new_all_in = state.all_in
            return (
                new_chips_in,
                new_round_raised,
                new_remaining,
                new_folded,
                new_all_in,
                new_not_raise_num,
            )

        # Process action
        (
            new_chips_in,
            new_round_raised,
            new_remaining_chips,
            new_folded,
            new_all_in,
            new_not_raise_num,
        ) = lax.switch(
            action,
            [
                process_check_call,
                process_raise_half_pot,
                process_raise_pot,
                process_all_in,
                process_fold,
            ],
        )

        # Count active (non-folded) players
        num_active_players = jnp.sum(~new_folded)
        num_playable_players = jnp.sum(~new_folded & ~new_all_in)

        # Check if betting round is over: all playable players have checked/called
        round_over = new_not_raise_num >= num_playable_players

        # Handle round transition: if round is over, move to next stage and reset betting state
        new_stage = jnp.where(round_over, state.stage + 1, state.stage)
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)
        new_round_raised = jnp.where(
            round_over, jnp.zeros_like(new_round_raised), new_round_raised
        )

        # Move to next player (skip folded players and all-in players)
        def get_next_player(
            start_idx: IntArray, folded_mask: BoolArray, all_in_mask: BoolArray
        ) -> IntArray:
            """Find next active player starting from start_idx"""

            # Use scan to check each possible next player position
            def check_player(carry, offset):
                idx = (start_idx + offset) % self.num_agents
                is_inactive = folded_mask[idx] | all_in_mask[idx]
                # If we haven't found a player yet and this one is active, use it
                found_idx, found = carry
                new_found_idx = jnp.where(~found & ~is_inactive, idx, found_idx)
                new_found = found | ~is_inactive
                return (new_found_idx, new_found), None

            # Check all players (at most num_agents iterations)
            offsets = jnp.arange(self.num_agents)
            (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
            return final_idx

        # Determine next player
        # If round is over, start new round from small blind (or first non-folded after small blind)
        # Otherwise, advance to next non-folded player
        start_idx = (current_player + 1) % self.num_agents
        next_player_after_action = get_next_player(start_idx, new_folded, new_all_in)

        next_player_new_round = get_next_player(
            state.small_blind_idx, new_folded, new_all_in
        )

        new_current_player_idx = jnp.where(
            round_over, next_player_new_round, next_player_after_action
        )

        # Check if game is done
        # Game ends if: only one player remains, no one can act, or all 4 betting rounds complete
        game_done = (
            (num_active_players <= 1) | (num_playable_players == 0) | (new_stage > 3)
        )

        # Update timestep
        new_timestep = state.timestep + 1

        # Create next state
        next_state = TexasNoLimitHoldEmState(
            flop_cards=state.flop_cards,
            turn_card=state.turn_card,
            river_card=state.river_card,
            agent_cards=state.agent_cards,
            chips_in=new_chips_in,
            round_raised=new_round_raised,
            remaining_chips=new_remaining_chips,
            small_blind_idx=state.small_blind_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            folded=new_folded,
            all_in=new_all_in,
            not_raise_num=new_not_raise_num,
            absorbing=jnp.broadcast_to(game_done, (self.num_agents,)),
            done=game_done,
            timestep=new_timestep,
        )

        # Get observation for next player
        obs = self.obs_from_state(next_state)

        # Rewards: zeros during game, payoffs calculated at end
        def compute_rewards() -> FloatArray:
            public_cards = jnp.concatenate(
                [state.flop_cards, jnp.array([state.turn_card, state.river_card])]
            )
            public_cards_tiled = jnp.tile(public_cards, (self.num_agents, 1))
            all_hands = jnp.concatenate([state.agent_cards, public_cards_tiled], axis=1)

            winners_by_score = _compare_hands(all_hands, new_folded)
            winners = jnp.where(num_active_players == 1, ~new_folded, winners_by_score)

            pot_total = jnp.sum(new_chips_in).astype(jnp.float32)
            num_winners = jnp.sum(winners).astype(jnp.float32)
            chips_gained = jnp.where(winners, pot_total / num_winners, 0.0)
            payoffs = chips_gained - new_chips_in.astype(jnp.float32)
            return payoffs

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents),
        )

        # Absorbing and done flags
        absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        done = absorbing.all()
        info = {"timestep": new_timestep}

        return next_state, obs, rewards, absorbing, done, info
