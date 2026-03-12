"""
5-Card Draw poker environment.
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
from bluffjax.utils.game_utils.poker_utils import _compare_five_card_hands


@struct.dataclass
class FiveCardDrawState:
    agent_cards: IntArray  # (num_agents, 5) - each player's 5-card hand
    chips_in: FloatArray
    round_raised: FloatArray
    remaining_chips: FloatArray
    small_blind_idx: IntArray
    current_player_idx: IntArray
    stage: IntArray  # 0=pre-draw betting, 1=draw, 2=post-draw betting
    folded: BoolArray
    all_in: BoolArray
    not_raise_num: IntArray
    shuffled_deck: IntArray  # (52,) - full deck for replacements
    deck_idx: IntArray
    draw_start_idx: IntArray  # First player in draw phase (for detecting completion)
    absorbing: BoolArray
    done: bool
    timestep: int


class FiveCardDraw(AECEnv):
    def __init__(
        self, num_agents: int = 2, horizon: int = 100_000, init_chips: int = 100
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 52
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind
        self.init_chips = init_chips
        self.obs_dim = self.deck_size + 2
        self.num_actions = (
            37  # 0-4 betting, 5-36 draw (2^5 binary keep/discard patterns)
        )

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: FiveCardDrawState) -> FloatArray:
        """
        Observation format (54 dimensions) - same as no-limit:
        - 0-51: One-hot encoding of current player's 5 cards
        - 52: Current player's chips in pot
        - 53: Max chips in pot
        """
        hand_cards = state.agent_cards[state.current_player_idx]
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        obs = obs.at[hand_cards].set(1.0)
        obs = obs.at[52].set(
            state.chips_in[state.current_player_idx].astype(jnp.float32)
        )
        obs = obs.at[53].set(jnp.max(state.chips_in).astype(jnp.float32))
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: FiveCardDrawState) -> BoolArray:
        """
        Actions: 0-4 betting (check/call, raise half, raise pot, all in, fold),
                5-36 draw: action 5+k = binary pattern k for keep/discard.
                Bit j=0 (LSB) = card at position 0 (smallest), bit 4 = position 4 (largest).
                0=discard, 1=keep. So action 5 = [00000] = discard all, action 36 = [11111] = keep all.
        """
        avail_actions = jnp.zeros(self.num_actions, dtype=bool)

        # Betting stages (0 and 2)
        is_betting = (state.stage == 0) | (state.stage == 2)
        current_player = state.current_player_idx
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)
        diff = max_round - player_round

        can_raise = player_remain > diff
        can_raise_pot = can_raise & (pot <= player_remain)
        can_raise_half = (
            can_raise
            & (half_pot <= player_remain)
            & ((half_pot + player_round) > max_round)
        )

        avail_actions = avail_actions.at[0].set(is_betting)
        avail_actions = avail_actions.at[1].set(is_betting & can_raise_half)
        avail_actions = avail_actions.at[2].set(is_betting & can_raise_pot)
        avail_actions = avail_actions.at[3].set(is_betting & can_raise)
        avail_actions = avail_actions.at[4].set(is_betting)

        # Draw stage (stage 1) - all 32 binary patterns valid for current player
        is_draw = state.stage == 1
        in_draw = is_draw & ~state.folded[current_player]
        avail_actions = avail_actions.at[5:37].set(in_draw)

        return avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[FiveCardDrawState, FloatArray]:
        """Return the first state of the game"""
        rng_shuffle, rng_player = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)
        agent_cards = shuffled_deck[: 5 * self.num_agents].reshape(-1, 5)
        deck_idx = jnp.int32(5 * self.num_agents)

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

        state = FiveCardDrawState(
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
            shuffled_deck=shuffled_deck,
            deck_idx=deck_idx,
            draw_start_idx=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.array(False),
            timestep=0,
        )

        obs = self.obs_from_state(state)
        return state, obs

    def _get_next_player_folded(
        self, start_idx: IntArray, folded_mask: BoolArray
    ) -> IntArray:
        """Find next non-folded player starting from (start_idx + 1) % num_agents."""

        def check_player(carry, offset):
            idx = (start_idx + 1 + offset) % self.num_agents
            is_folded = folded_mask[idx]
            found_idx, found = carry
            new_found_idx = jnp.where(~found & ~is_folded, idx, found_idx)
            new_found = found | ~is_folded
            return (new_found_idx, new_found), None

        offsets = jnp.arange(self.num_agents)
        (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
        return final_idx

    def _get_next_player(
        self, start_idx: IntArray, folded_mask: BoolArray, all_in_mask: BoolArray
    ) -> IntArray:
        """Find next active (non-folded, non-all-in) player."""

        def check_player(carry, offset):
            idx = (start_idx + 1 + offset) % self.num_agents
            is_inactive = folded_mask[idx] | all_in_mask[idx]
            found_idx, found = carry
            new_found_idx = jnp.where(~found & ~is_inactive, idx, found_idx)
            new_found = found | ~is_inactive
            return (new_found_idx, new_found), None

        offsets = jnp.arange(self.num_agents)
        (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
        return final_idx

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: FiveCardDrawState, action: IntArray
    ) -> tuple[
        FiveCardDrawState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step. Actions: 0-4 betting, 5-36 draw (2^5 binary keep/discard).
        """
        current_player = state.current_player_idx
        stage = state.stage

        # --- Betting logic (stage 0 or 2) ---
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)
        diff = max_round - player_round

        def apply_bet(bet_amount: FloatArray):
            new_chips_in = state.chips_in.at[current_player].add(bet_amount)
            new_round_raised = state.round_raised.at[current_player].add(bet_amount)
            new_remaining = state.remaining_chips.at[current_player].add(-bet_amount)
            new_all_in = state.all_in.at[current_player].set(
                new_remaining[current_player] <= 0
            )
            return new_chips_in, new_round_raised, new_remaining, new_all_in

        def process_check_call():
            nc, nr, nrem, na = apply_bet(diff)
            all_in = na[current_player]
            nnr = jnp.where(all_in, state.not_raise_num, state.not_raise_num + 1)
            return nc, nr, nrem, state.folded, na, nnr

        def process_raise_half_pot():
            nc, nr, nrem, na = apply_bet(half_pot)
            nnr = jnp.int32(1)
            return nc, nr, nrem, state.folded, na, nnr

        def process_raise_pot():
            nc, nr, nrem, na = apply_bet(pot)
            nnr = jnp.int32(1)
            return nc, nr, nrem, state.folded, na, nnr

        def process_all_in():
            nc, nr, nrem, na = apply_bet(player_remain)
            nnr = jnp.int32(1)
            return nc, nr, nrem, state.folded, na, nnr

        def process_fold():
            nf = state.folded.at[current_player].set(True)
            return (
                state.chips_in,
                state.round_raised,
                state.remaining_chips,
                nf,
                state.all_in,
                state.not_raise_num,
            )

        (nc, nr, nrem, nf, na, nnr) = lax.switch(
            action,
            [
                process_check_call,
                process_raise_half_pot,
                process_raise_pot,
                process_all_in,
                process_fold,
            ],
        )

        # Use betting result only when in betting stage
        is_betting = (stage == 0) | (stage == 2)
        new_chips_in = jnp.where(is_betting, nc, state.chips_in)
        new_round_raised = jnp.where(is_betting, nr, state.round_raised)
        new_remaining_chips = jnp.where(is_betting, nrem, state.remaining_chips)
        new_folded = jnp.where(is_betting, nf, state.folded)
        new_all_in = jnp.where(is_betting, na, state.all_in)
        new_not_raise_num = jnp.where(is_betting, nnr, state.not_raise_num)

        num_active = jnp.sum(~new_folded)
        num_playable = jnp.sum(~new_folded & ~new_all_in)
        round_over = new_not_raise_num >= num_playable

        # --- Draw logic (stage 1) ---
        hand = state.agent_cards[current_player]
        sorted_hand = jnp.sort(hand)

        def apply_draw_replacements():
            # Action 5+k: binary pattern k. Bit j (LSB=j=0) = card at sorted position j.
            # 0=discard, 1=keep. So discard_mask[j] = 1 - ((action-5) >> j) & 1
            draw_pattern = action - 5
            powers = 2 ** jnp.arange(5, dtype=jnp.int32)
            keep_bits = (draw_pattern // powers) % 2
            mask = (1 - keep_bits).astype(bool)  # True = discard

            num_discard = jnp.sum(mask).astype(jnp.int32)
            all_hand_marked = jnp.where(mask, 999, sorted_hand)
            hand_sorted = jnp.sort(all_hand_marked)
            deck = state.shuffled_deck

            def replace_n(n: int):
                num_kept = 5 - n
                kept_cards = hand_sorted[:num_kept]
                new_cards = lax.dynamic_slice(deck, (state.deck_idx,), (n,))
                new_hand = jnp.sort(jnp.concatenate([kept_cards, new_cards]))
                new_agent_cards = state.agent_cards.at[current_player].set(new_hand)
                new_deck_idx = state.deck_idx + n
                return new_agent_cards, new_deck_idx

            return lax.switch(
                num_discard,
                [
                    lambda: replace_n(0),
                    lambda: replace_n(1),
                    lambda: replace_n(2),
                    lambda: replace_n(3),
                    lambda: replace_n(4),
                    lambda: replace_n(5),
                ],
            )

        # Draw: action 5-36 applies the binary keep/discard pattern and advances
        did_apply_draw = (
            (stage == 1)
            & (action >= 5)
            & (action <= 36)
            & ~state.folded[current_player]
        )
        new_agent_cards, new_deck_idx = lax.cond(
            did_apply_draw,
            apply_draw_replacements,
            lambda: (state.agent_cards, state.deck_idx),
        )

        # When we apply draw, advance to next non-folded player
        next_draw_player = self._get_next_player_folded(current_player, state.folded)
        draw_phase_done = next_draw_player == state.draw_start_idx
        post_draw_start = self._get_next_player(
            state.small_blind_idx, new_folded, new_all_in
        )

        new_current_player_idx = jnp.where(
            did_apply_draw & draw_phase_done,
            post_draw_start,
            jnp.where(did_apply_draw, next_draw_player, current_player),
        )

        # Stay in draw or move to stage 2
        new_stage = jnp.where(
            did_apply_draw & draw_phase_done,
            jnp.int32(2),
            jnp.where(did_apply_draw, stage, stage),
        )

        # --- Betting round transition ---
        # When pre-draw betting ends: go to draw (stage 1) or game_done if 1 player
        pre_draw_ended = (stage == 0) & round_over
        skip_draw = num_active <= 1
        to_draw_stage = pre_draw_ended & ~skip_draw
        to_stage2_direct = pre_draw_ended & skip_draw

        draw_start = self._get_next_player_folded(state.small_blind_idx, new_folded)

        new_stage = jnp.where(
            to_draw_stage,
            jnp.int32(1),
            jnp.where(
                to_stage2_direct,
                jnp.int32(2),
                new_stage,
            ),
        )

        # Reset betting when round over
        new_round_raised = jnp.where(
            round_over,
            jnp.zeros_like(new_round_raised),
            new_round_raised,
        )
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)

        # When entering draw: set current_player, draw_start
        new_current_player_idx = jnp.where(
            to_draw_stage, draw_start, new_current_player_idx
        )
        new_draw_start_idx = jnp.where(to_draw_stage, draw_start, state.draw_start_idx)

        # When pre-draw ends with 1 player, go to showdown
        new_current_player_idx = jnp.where(
            to_stage2_direct, current_player, new_current_player_idx
        )

        # Next player for betting (when round not over)
        next_bet_player = self._get_next_player(current_player, new_folded, new_all_in)
        next_bet_player_new_round = self._get_next_player(
            state.small_blind_idx, new_folded, new_all_in
        )
        new_current_player_idx = jnp.where(
            is_betting & ~round_over,
            next_bet_player,
            jnp.where(
                is_betting & round_over & (stage == 2),
                next_bet_player_new_round,
                new_current_player_idx,
            ),
        )

        # Game done
        game_done = (
            (num_active <= 1)
            | (num_playable == 0)
            | (to_stage2_direct)
            | ((stage == 2) & round_over)
        )

        def compute_rewards():
            winners = _compare_five_card_hands(state.agent_cards, new_folded)
            winners = jnp.where(num_active <= 1, ~new_folded, winners)
            pot_total = jnp.sum(new_chips_in).astype(jnp.float32)
            num_winners = jnp.sum(winners).astype(jnp.float32)
            chips_gained = jnp.where(winners, pot_total / num_winners, 0.0)
            return chips_gained - new_chips_in.astype(jnp.float32)

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents),
        )

        next_state = FiveCardDrawState(
            agent_cards=new_agent_cards,
            chips_in=new_chips_in,
            round_raised=new_round_raised,
            remaining_chips=new_remaining_chips,
            small_blind_idx=state.small_blind_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            folded=new_folded,
            all_in=new_all_in,
            not_raise_num=new_not_raise_num,
            shuffled_deck=state.shuffled_deck,
            deck_idx=new_deck_idx,
            draw_start_idx=new_draw_start_idx,
            absorbing=jnp.broadcast_to(game_done, (self.num_agents,)),
            done=game_done,
            timestep=state.timestep + 1,
        )

        obs = self.obs_from_state(next_state)
        absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        done = absorbing.all()
        game_winner = (rewards > 0.0).astype(jnp.float32)
        info = {
            "returns": rewards.astype(jnp.float32),
            "timestep": next_state.timestep,
            "game_winner": game_winner,
        }
        return next_state, obs, rewards, absorbing, done, info
