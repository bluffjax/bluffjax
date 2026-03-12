"""
Texas Limit Hold'em environment.
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
from bluffjax.utils.game_utils.poker_utils import _compare_hands


@struct.dataclass
class TexasLimitHoldEmState:
    flop_cards: FloatArray
    turn_card: FloatArray
    river_card: FloatArray
    agent_cards: FloatArray
    chips_in: FloatArray
    small_blind_idx: IntArray
    current_player_idx: IntArray
    stage: IntArray  # 0=preflop, 1=flop, 2=turn, 3=river
    raise_nums: IntArray  # Number of raises per betting round (4 rounds)
    folded: BoolArray  # Whether each player has folded
    not_raise_num: IntArray  # Number of players who have checked/called without raising
    absorbing: BoolArray
    done: bool
    timestep: int


class TexasLimitHoldem(AECEnv):
    def __init__(self, num_agents: int = 2, horizon: int = 100_000) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 52
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind
        self.raise_amount = self.big_blind
        self.allowed_raise_num = 4
        self.base_obs_dim = 72  # 52 cards + 20 raise history (4 rounds * 5 values)
        self.obs_dim = 72 + num_agents  # base + one-hot position encoding
        self.num_betting_rounds = 4
        self.num_actions = 4  # call, raise, fold, check

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: TexasLimitHoldEmState) -> FloatArray:
        """
        Extract observation for a specific agent in rlcard limitholdem style.

        Observation format (72 + num_agents dimensions):
        - Dimensions 0-51: One-hot encoding of cards (hand cards + public cards)
        - Dimensions 52-71: One-hot encoding of raise counts per betting round
          (4 rounds *5 possible values: 0-4 raises)
        - Dimensions 72-(72+num_agents-1): One-hot encoding of relative position
          to small blind player (small blind = index 0)

        Args:
            state: Current game state

        Returns:
            observation: (72 + num_agents)-dimensional observation vector
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

        # Initialize observation vector (72 + num_agents dimensions)
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        obs = obs.at[hand_cards].set(1.0)

        obs = jnp.where(stage >= 1, obs.at[flop].set(1.0), obs)
        obs = jnp.where(stage >= 2, obs.at[turn].set(1.0), obs)
        obs = jnp.where(stage >= 3, obs.at[river].set(1.0), obs)

        def encode_raise_round(obs, stage):
            obs_idx = 52 + stage * 5 + state.raise_nums[stage]
            return obs.at[obs_idx].set(1.0)

        obs = encode_raise_round(obs, 0)
        obs = encode_raise_round(obs, 1)
        obs = encode_raise_round(obs, 2)
        obs = encode_raise_round(obs, 3)

        relative_position = (
            state.current_player_idx - state.small_blind_idx
        ) % self.num_agents
        position_idx = self.base_obs_dim + relative_position
        obs = obs.at[position_idx].set(1.0)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: TexasLimitHoldEmState) -> BoolArray:
        """
        Get available actions mask for the current player.

        Actions (in order):
        0: call
        1: raise
        2: fold
        3: check

        Rules:
        - Raise: Only allowed if current_round_raises < allowed_raise_num (4)
        - Fold: Always allowed
        - Call: Not allowed if player's chips_in == max(chips_in)
        - Check: Not allowed if player's chips_in < max(chips_in)

        Args:
            state: Current game state

        Returns:
            avail_actions: Boolean mask of shape (4,) indicating which actions are legal
        """

        # When game is done, return all zeros to avoid indexing issues with stage >= 4
        def compute_avail() -> BoolArray:
            current_player = state.current_player_idx
            player_chips = state.chips_in[current_player]
            max_chips = jnp.max(state.chips_in)
            avail_actions = jnp.ones(4, dtype=bool)
            can_raise = state.raise_nums[state.stage] < self.allowed_raise_num
            avail_actions = avail_actions.at[1].set(can_raise)
            can_call = player_chips < max_chips
            avail_actions = avail_actions.at[0].set(can_call)
            can_check = player_chips == max_chips
            avail_actions = avail_actions.at[3].set(can_check)
            return avail_actions

        return lax.cond(state.done, lambda: jnp.zeros(4, dtype=bool), compute_avail)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> TexasLimitHoldEmState:
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
        current_player_idx = (small_blind_agent + 2) % self.num_agents

        # Initialize raise_nums: track raises per betting round (4 rounds)
        raise_nums = jnp.zeros(self.num_betting_rounds, dtype=jnp.int32)
        folded = jnp.zeros(self.num_agents, dtype=bool)
        not_raise_num = jnp.int32(0)  # Will be updated as players act

        state = TexasLimitHoldEmState(
            flop_cards=flop,
            turn_card=turn,
            river_card=river,
            agent_cards=agent_cards,
            chips_in=chips_in,
            small_blind_idx=small_blind_agent,
            current_player_idx=current_player_idx,
            stage=jnp.int32(0),
            raise_nums=raise_nums,
            folded=folded,
            not_raise_num=not_raise_num,
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.array(False),
            timestep=0,
        )

        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: TexasLimitHoldEmState, action: IntArray
    ) -> tuple[
        TexasLimitHoldEmState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step in the game.

        Args:
            state: Current game state
            action: Action taken by current player (0=call, 1=raise, 2=fold, 3=check)

        Returns:
            next_state: Updated game state
            obs: Observation for next player
            reward: Rewards for all players (zeros during game, payoffs at end)
            absorbing: Whether absorbing state reached
            done: Whether game is done
        """
        current_player = state.current_player_idx
        max_chips = jnp.max(state.chips_in)
        player_chips = state.chips_in[current_player]

        # Get raise amount for current stage (doubles after flop)
        raise_amount = jnp.where(
            state.stage >= 2, self.raise_amount * 2, self.raise_amount
        )

        # Get current round raise count
        current_round_raises = state.raise_nums[state.stage]

        # Process action using switch
        def process_call() -> tuple[FloatArray, IntArray, IntArray, BoolArray]:
            diff = max_chips - player_chips
            new_chips_in = state.chips_in.at[current_player].add(diff)
            new_raise_nums = state.raise_nums  # No change to raise count
            new_not_raise_num = state.not_raise_num + 1
            new_folded = state.folded
            return new_chips_in, new_raise_nums, new_not_raise_num, new_folded

        def process_raise() -> tuple[FloatArray, IntArray, IntArray, BoolArray]:
            diff = max_chips - player_chips + raise_amount
            new_chips_in = state.chips_in.at[current_player].add(diff)
            # Increment raise count for current stage
            new_raise_nums = state.raise_nums.at[state.stage].set(
                current_round_raises + 1
            )
            new_not_raise_num = jnp.int32(1)  # Reset to 1 after raise
            new_folded = state.folded
            return new_chips_in, new_raise_nums, new_not_raise_num, new_folded

        def process_fold() -> tuple[FloatArray, IntArray, IntArray, BoolArray]:
            new_chips_in = state.chips_in
            new_raise_nums = state.raise_nums
            new_not_raise_num = state.not_raise_num
            new_folded = state.folded.at[current_player].set(True)
            return new_chips_in, new_raise_nums, new_not_raise_num, new_folded

        def process_check() -> tuple[FloatArray, IntArray, IntArray, BoolArray]:
            new_chips_in = state.chips_in
            new_raise_nums = state.raise_nums
            new_not_raise_num = state.not_raise_num + 1
            new_folded = state.folded
            return new_chips_in, new_raise_nums, new_not_raise_num, new_folded

        # Process action
        new_chips_in, new_raise_nums, new_not_raise_num, new_folded = lax.switch(
            action, [process_call, process_raise, process_fold, process_check]
        )

        # Count active (non-folded) players
        num_active_players = jnp.sum(~new_folded)

        # Check if betting round is over: all active players have checked/called without raising
        round_over = new_not_raise_num >= num_active_players

        # Handle round transition: if round is over, move to next stage and reset betting state
        new_stage = jnp.where(round_over, state.stage + 1, state.stage)
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)

        # Move to next player (skip folded players)
        def get_next_player(current_idx: IntArray, folded_mask: BoolArray) -> IntArray:
            """Find next non-folded player starting from (current_idx + 1) % num_agents"""
            start_idx = (current_idx + 1) % self.num_agents

            # Use scan to check each possible next player position
            def check_player(carry, offset):
                idx = (start_idx + offset) % self.num_agents
                is_folded = folded_mask[idx]
                # If we haven't found a player yet and this one isn't folded, use it
                found_idx, found = carry
                new_found_idx = jnp.where(~found & ~is_folded, idx, found_idx)
                new_found = found | ~is_folded
                return (new_found_idx, new_found), None

            # Check all players (at most num_agents iterations)
            offsets = jnp.arange(self.num_agents)
            (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
            return final_idx

        # Determine next player
        # If round is over, start new round from small blind (or first non-folded after small blind)
        # Otherwise, advance to next non-folded player
        next_player_after_action = get_next_player(current_player, new_folded)

        next_player_new_round = jnp.where(
            ~new_folded[state.small_blind_idx],
            state.small_blind_idx,
            get_next_player(state.small_blind_idx, new_folded),
        )

        new_current_player_idx = jnp.where(
            round_over, next_player_new_round, next_player_after_action
        )

        # Check if game is done
        # Game ends if: only one player remains, or all 4 betting rounds complete
        game_done = (num_active_players <= 1) | (new_stage >= 4)

        # Update timestep
        new_timestep = state.timestep + 1

        # Create next state
        next_state = TexasLimitHoldEmState(
            flop_cards=state.flop_cards,
            turn_card=state.turn_card,
            river_card=state.river_card,
            agent_cards=state.agent_cards,
            chips_in=new_chips_in,
            small_blind_idx=state.small_blind_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            raise_nums=new_raise_nums,
            folded=new_folded,
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
            payoffs = (chips_gained - new_chips_in.astype(jnp.float32)) / self.big_blind
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
