"""
7-Card Stud poker environment.
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
from bluffjax.utils.game_utils.poker_utils import (
    _compare_hands,
    _get_bring_in_idx,
    _score_visible_upcards,
)


@struct.dataclass
class SevenCardStudState:
    agent_cards: IntArray  # (num_agents, 7) - all 7 cards per player
    chips_in: FloatArray  # (num_agents,) - per-player contribution this hand
    bring_in_idx: IntArray  # who posts bring-in (3rd street)
    current_player_idx: IntArray
    stage: IntArray  # 0=3rd, 1=4th, 2=5th, 3=6th, 4=7th street
    raise_nums: IntArray  # (5,) - raises per betting round
    folded: BoolArray
    not_raise_num: IntArray
    absorbing: BoolArray
    done: bool
    timestep: int


# Visibility: stage 0 -> indices [2], stage 1 -> [2,3], stage 2 -> [2,3,4], stage 3,4 -> [2,3,4,5]
NUM_VISIBLE_BY_STAGE = jnp.array([1, 2, 3, 4, 4], dtype=jnp.int32)


class SevenCardStud(AECEnv):
    def __init__(
        self, num_agents: int = 2, horizon: int = 100_000, init_chips: int = 100
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 52
        self.small_bet = 1
        self.big_bet = 2
        self.ante = 0.5
        self.bring_in = 0.5
        self.raise_amount_small = self.small_bet
        self.raise_amount_big = self.big_bet
        self.allowed_raise_num = 4
        self.init_chips = init_chips
        self.num_betting_rounds = 5
        # obs: 52 own + (num_agents-1)*4*52 others' visible + 25 raise history + num_agents position
        self.base_obs_dim = 52 + (num_agents - 1) * 4 * 52 + 25
        self.obs_dim = self.base_obs_dim + num_agents

    @partial(jax.jit, static_argnums=(0,))
    def _get_visible_slice(self, stage: IntArray) -> IntArray:
        """Returns end index for visible cards: stage 0->1, 1->2, 2->3, 3->4, 4->4."""
        return NUM_VISIBLE_BY_STAGE[stage] + 2  # 2 is start index

    @partial(jax.jit, static_argnums=(0,))
    def _get_first_to_act_after_bring_in(
        self, bring_in_idx: IntArray, folded: BoolArray
    ) -> IntArray:
        """First non-folded player after bring-in (clockwise)."""
        start_idx = (bring_in_idx + 1) % self.num_agents

        def check_player(carry, offset):
            idx = (start_idx + offset) % self.num_agents
            is_folded = folded[idx]
            found_idx, found = carry
            new_found_idx = jnp.where(~found & ~is_folded, idx, found_idx)
            new_found = found | ~is_folded
            return (new_found_idx, new_found), None

        offsets = jnp.arange(self.num_agents)
        (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
        return final_idx

    @partial(jax.jit, static_argnums=(0,))
    def _get_first_to_act_by_upcards(
        self,
        agent_cards: IntArray,
        folded: BoolArray,
        stage: IntArray,
    ) -> IntArray:
        """Player with best visible upcards acts first. Stage 1->2 cards, 2->3, 3,4->4."""
        stage_clamped = jnp.minimum(stage, 4)

        visible_2 = agent_cards[:, 2:4]  # 2 cards
        visible_3 = agent_cards[:, 2:5]  # 3 cards
        visible_4 = agent_cards[:, 2:6]  # 4 cards

        scores_2 = jax.vmap(_score_visible_upcards)(visible_2)
        scores_3 = jax.vmap(_score_visible_upcards)(visible_3)
        scores_4 = jax.vmap(_score_visible_upcards)(visible_4)

        scores = jnp.where(
            stage_clamped == 1,
            scores_2,
            jnp.where(
                stage_clamped == 2,
                scores_3,
                scores_4,
            ),
        )
        # Folded players get -1 so they never win
        scores = jnp.where(folded, -1, scores)
        return jnp.argmax(scores)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: SevenCardStudState) -> FloatArray:
        """
        Observation from perspective of current_player_idx.
        - Own 7 cards (one-hot 0-51)
        - Others' visible cards in relative order: (current+1)%n, (current+2)%n, ...
        - Raise history (5 rounds * 5 values)
        - Position relative to bring-in
        """
        current = state.current_player_idx
        stage = state.stage
        num_visible = NUM_VISIBLE_BY_STAGE[stage]

        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)

        # Own cards (all 7)
        own_cards = state.agent_cards[current]
        obs = obs.at[own_cards].set(1.0)

        # Others' visible cards in relative order (always 4 slots per other, mask by num_visible)
        base = 52
        for i in range(self.num_agents - 1):
            other_idx = (current + 1 + i) % self.num_agents
            other_visible = state.agent_cards[other_idx, 2:6]  # indices 2,3,4,5
            for j in range(4):
                include = jnp.int32(j) < num_visible
                idx = base + i * 4 * 52 + j * 52 + other_visible[j]
                new_obs = obs.at[idx].set(1.0)
                obs = jnp.where(include, new_obs, obs)

        # Raise history
        raise_base = 52 + (self.num_agents - 1) * 4 * 52
        for s in range(5):
            obs = obs.at[raise_base + s * 5 + state.raise_nums[s]].set(1.0)

        # Position relative to bring-in
        rel_pos = (current - state.bring_in_idx) % self.num_agents
        obs = obs.at[self.base_obs_dim + rel_pos].set(1.0)

        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: SevenCardStudState) -> BoolArray:
        """
        Actions: 0=call, 1=raise, 2=fold, 3=check
        Same logic as Texas Limit Hold'em.
        """
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

    @partial(jax.jit, static_argnums=(0,))
    def avail_actions(self, state: SevenCardStudState) -> BoolArray:
        """AECEnv interface: returns available actions for current player."""
        return self.get_avail_actions(state)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[SevenCardStudState, FloatArray]:
        """Reset: deal all 7 cards, compute bring-in, post ante and bring-in."""
        rng_shuffle, _ = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)

        # Deal 7 cards per player (all predetermined)
        agent_cards = shuffled_deck[: 7 * self.num_agents].reshape(self.num_agents, 7)

        # Bring-in: lowest door card (index 2)
        door_cards = agent_cards[:, 2]
        bring_in_idx = _get_bring_in_idx(door_cards)

        # Ante: each player posts ante; bring-in: bring_in_idx posts bring-in
        chips_in = jnp.full(self.num_agents, self.ante, dtype=jnp.float32)
        chips_in = chips_in.at[bring_in_idx].add(self.bring_in)

        # First to act: first non-folded after bring-in (3rd street)
        folded = jnp.zeros(self.num_agents, dtype=bool)
        current_player_idx = self._get_first_to_act_after_bring_in(bring_in_idx, folded)

        raise_nums = jnp.zeros(self.num_betting_rounds, dtype=jnp.int32)
        not_raise_num = jnp.int32(0)

        state = SevenCardStudState(
            agent_cards=agent_cards,
            chips_in=chips_in,
            bring_in_idx=bring_in_idx,
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
        self, rng: PRNGKeyArray, state: SevenCardStudState, action: IntArray
    ) -> tuple[
        SevenCardStudState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step. Actions: 0=call, 1=raise, 2=fold, 3=check.
        """
        current_player = state.current_player_idx
        max_chips = jnp.max(state.chips_in)
        player_chips = state.chips_in[current_player]

        # Raise amount: small bet for stage 0,1; big bet for stage 2,3,4
        raise_amount = jnp.where(
            state.stage >= 2, self.raise_amount_big, self.raise_amount_small
        )
        current_round_raises = state.raise_nums[state.stage]

        def process_call():
            diff = max_chips - player_chips
            new_chips_in = state.chips_in.at[current_player].add(diff)
            return new_chips_in, state.raise_nums, state.not_raise_num + 1, state.folded

        def process_raise():
            diff = max_chips - player_chips + raise_amount
            new_chips_in = state.chips_in.at[current_player].add(diff)
            new_raise_nums = state.raise_nums.at[state.stage].set(
                current_round_raises + 1
            )
            return new_chips_in, new_raise_nums, jnp.int32(1), state.folded

        def process_fold():
            new_folded = state.folded.at[current_player].set(True)
            return state.chips_in, state.raise_nums, state.not_raise_num, new_folded

        def process_check():
            return (
                state.chips_in,
                state.raise_nums,
                state.not_raise_num + 1,
                state.folded,
            )

        new_chips_in, new_raise_nums, new_not_raise_num, new_folded = lax.switch(
            action, [process_call, process_raise, process_fold, process_check]
        )

        num_active_players = jnp.sum(~new_folded)
        round_over = new_not_raise_num >= num_active_players

        new_stage = jnp.where(round_over, state.stage + 1, state.stage)
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)

        def get_next_player(current_idx: IntArray, folded_mask: BoolArray) -> IntArray:
            start_idx = (current_idx + 1) % self.num_agents

            def check_player(carry, offset):
                idx = (start_idx + offset) % self.num_agents
                is_folded = folded_mask[idx]
                found_idx, found = carry
                new_found_idx = jnp.where(~found & ~is_folded, idx, found_idx)
                new_found = found | ~is_folded
                return (new_found_idx, new_found), None

            offsets = jnp.arange(self.num_agents)
            (final_idx, _), _ = lax.scan(check_player, (start_idx, False), offsets)
            return final_idx

        # First to act for new round: when round_over, we're moving to next stage.
        # new_stage is always >= 1 when round_over (we never round_over into stage 0).
        # 4th-7th street: player with best visible upcards acts first.
        first_to_act_upcards = self._get_first_to_act_by_upcards(
            state.agent_cards, new_folded, new_stage
        )
        next_player_new_round = jnp.where(
            ~new_folded[first_to_act_upcards],
            first_to_act_upcards,
            get_next_player(first_to_act_upcards, new_folded),
        )

        next_player_after_action = get_next_player(current_player, new_folded)

        new_current_player_idx = jnp.where(
            round_over, next_player_new_round, next_player_after_action
        )

        game_done = (num_active_players <= 1) | (new_stage >= 5)

        next_state = SevenCardStudState(
            agent_cards=state.agent_cards,
            chips_in=new_chips_in,
            bring_in_idx=state.bring_in_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            raise_nums=new_raise_nums,
            folded=new_folded,
            not_raise_num=new_not_raise_num,
            absorbing=jnp.broadcast_to(game_done, (self.num_agents,)),
            done=game_done,
            timestep=state.timestep + 1,
        )

        obs = self.obs_from_state(next_state)

        def compute_rewards() -> FloatArray:
            winners_by_score = _compare_hands(state.agent_cards, new_folded)
            winners = jnp.where(num_active_players == 1, ~new_folded, winners_by_score)
            pot_total = jnp.sum(new_chips_in).astype(jnp.float32)
            num_winners = jnp.sum(winners).astype(jnp.float32)
            chips_gained = jnp.where(winners, pot_total / num_winners, 0.0)
            payoffs = (chips_gained - new_chips_in.astype(jnp.float32)) / self.big_bet
            return payoffs

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents),
        )

        absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        done = absorbing.all()
        game_winner = (rewards > 0.0).astype(jnp.float32)
        info = {
            "returns": rewards.astype(jnp.float32),
            "timestep": next_state.timestep,
            "game_winner": game_winner,
        }
        return next_state, obs, rewards, absorbing, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: PRNGKeyArray,
        state: SevenCardStudState,
        action: IntArray,
    ) -> tuple[
        SevenCardStudState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """AEC step: run step_env, reset on done."""
        rng_step, rng_reset = jax.random.split(rng)
        state_next, obs, rewards, absorbing, done, info = self.step_env(
            rng_step, state, action
        )
        state_reset, obs_reset = self.reset(rng_reset)
        state_final = lax.cond(done, lambda: state_reset, lambda: state_next)
        obs_final = lax.cond(done, lambda: obs_reset, lambda: obs)
        return state_final, obs_final, rewards, absorbing, done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(4)
