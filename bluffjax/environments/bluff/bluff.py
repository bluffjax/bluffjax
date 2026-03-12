"""
Bluff environment (AEC).
This code models the dynamics of the game 'Bluff' AKA Cheat, I Doubt It:
https://en.wikipedia.org/wiki/Cheat_(game).
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax.environments.env import AECEnv
from bluffjax.environments.spaces import Discrete


@struct.dataclass
class BluffState:
    pile_hand: FloatArray
    pile_claims: FloatArray
    pile_size: IntArray
    agent_hands: FloatArray
    agent_hand_sizes: IntArray
    phase: IntArray  # 0=claim, 1=play, 2=challenge
    current_player_idx: IntArray
    start_player_idx: IntArray
    round_start_player_idx: IntArray
    challenge_target_idx: IntArray
    claim_size: IntArray
    claim_rank: IntArray
    pending_play_hand: FloatArray
    pending_play_count: IntArray
    has_current_rank: BoolArray
    current_rank: IntArray
    rank_choice_pending: BoolArray
    challenge_status: IntArray  # 0=target won, 1=challenger won, 2=no challenge, 3=none
    challenge_hand: FloatArray
    absorbing: BoolArray
    done: bool
    game_winner: BoolArray
    timestep: int


class Bluff(AECEnv):
    def __init__(
        self,
        num_agents: int = 3,
        num_decks: int = 1,
        num_ranks: int = 13,
        num_suits: int = 4,
        horizon: int = 100,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.cards_per_rank = num_suits * num_decks
        self.deck_size = num_suits * num_decks * num_ranks
        self.cards_per_deck = num_suits * num_ranks
        self.action_dim = 13
        self.obs_dim = (
            2 * (self.cards_per_rank * num_ranks) + 3 * self.deck_size + 3
        )  # pile_claims + pile_size + own_hand + claimant_size + claim_size + phase_oh

        self._reward_card = 1.0
        self._reward_challenge = 1.0
        self._reward_win = 10.0

    @partial(jax.jit, static_argnums=(0,))
    def _empty_hand(self) -> FloatArray:
        return jnp.zeros((self.num_ranks,), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _empty_hands(self) -> FloatArray:
        return jnp.zeros((self.num_agents, self.num_ranks), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _encode_size_thermo(self, x: IntArray) -> FloatArray:
        return (jnp.arange(self.deck_size) < x).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _card_ints_to_hand(self, cards: IntArray) -> FloatArray:
        ranks = cards % self.cards_per_deck // self.num_suits
        return jnp.bincount(ranks, length=self.num_ranks).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _roll_if_rank_known(self, counts: FloatArray, state: BluffState) -> FloatArray:
        return lax.cond(
            state.has_current_rank,
            lambda: jnp.roll(counts, -state.current_rank, axis=0),
            lambda: counts,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _encode_suit_major_thermo(self, counts: FloatArray) -> FloatArray:
        suit_ids = jnp.arange(self.cards_per_rank)[:, None]
        # suit-major layout: [suit0: 13 ranks, suit1: 13 ranks, ...]
        return (suit_ids < counts[None, :]).astype(jnp.float32).reshape(-1)

    @partial(jax.jit, static_argnums=(0,))
    def _next_player(self, idx: IntArray) -> IntArray:
        return (idx + 1) % self.num_agents

    @partial(jax.jit, static_argnums=(0,))
    def _action_offset_to_rank(self, state: BluffState, action: IntArray) -> IntArray:
        return lax.cond(
            state.has_current_rank,
            lambda: (state.current_rank + action) % self.num_ranks,
            lambda: action % self.num_ranks,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: BluffState) -> FloatArray:
        player_idx = state.current_player_idx
        own_hand = state.agent_hands[player_idx]

        pile_claims_rel = self._roll_if_rank_known(state.pile_claims, state)
        pile_claims_obs = self._encode_suit_major_thermo(pile_claims_rel)

        own_hand_rel = self._roll_if_rank_known(own_hand, state)
        own_hand_obs = self._encode_suit_major_thermo(own_hand_rel)

        pile_size_obs = self._encode_size_thermo(state.pile_size)
        claimant_hand_size = state.agent_hand_sizes[state.challenge_target_idx]
        claimant_size_obs = self._encode_size_thermo(claimant_hand_size)
        claim_size_obs = self._encode_size_thermo(state.claim_size)

        challenge_phase = state.phase == 2
        claimant_size_obs = jnp.where(challenge_phase, claimant_size_obs, 0.0)
        claim_size_obs = jnp.where(challenge_phase, claim_size_obs, 0.0)

        phase_oh = jax.nn.one_hot(state.phase, 3, dtype=jnp.float32)
        return jnp.concatenate(
            [
                pile_claims_obs,
                pile_size_obs,
                own_hand_obs,
                claimant_size_obs,
                claim_size_obs,
                phase_oh,
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: BluffState) -> BoolArray:
        def claim_mask() -> BoolArray:
            hand_size = state.agent_hand_sizes[state.current_player_idx]
            max_claim = jnp.minimum(4, hand_size).astype(jnp.int32)
            return jnp.arange(self.action_dim) < max_claim

        def play_mask() -> BoolArray:
            action_ids = jnp.arange(self.action_dim)
            abs_ranks = lax.cond(
                state.has_current_rank,
                lambda: (state.current_rank + action_ids) % self.num_ranks,
                lambda: action_ids % self.num_ranks,
            )
            available = (
                state.agent_hands[state.current_player_idx, abs_ranks]
                - state.pending_play_hand[abs_ranks]
            ) > 0
            return available & (state.pending_play_count < state.claim_size)

        def challenge_mask() -> BoolArray:
            return jnp.arange(self.action_dim) < 2

        phase_masks = [claim_mask, play_mask, challenge_mask]
        return lax.cond(
            state.done,
            lambda: jnp.zeros((self.action_dim,), dtype=bool),
            lambda: lax.switch(state.phase, phase_masks),
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[BluffState, FloatArray]:
        rng_shuffle, rng_player = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)
        cards_per_player = self.deck_size // self.num_agents
        used_cards = self.num_agents * cards_per_player

        distributed = shuffled_deck[:used_cards].reshape(
            self.num_agents, cards_per_player
        )
        agent_hands = jax.vmap(self._card_ints_to_hand)(distributed)
        agent_hand_sizes = agent_hands.sum(axis=1).astype(jnp.int32)

        pile_cards_int = shuffled_deck[used_cards:]
        pile_hand = self._card_ints_to_hand(pile_cards_int)
        pile_size = pile_hand.sum().astype(jnp.int32)

        start_player = jax.random.randint(
            rng_player, shape=(), minval=0, maxval=self.num_agents
        ).astype(jnp.int32)

        state = BluffState(
            pile_hand=pile_hand,
            pile_claims=self._empty_hand(),
            pile_size=pile_size,
            agent_hands=agent_hands,
            agent_hand_sizes=agent_hand_sizes,
            phase=jnp.array(0, dtype=jnp.int32),
            current_player_idx=start_player,
            start_player_idx=start_player,
            round_start_player_idx=start_player,
            challenge_target_idx=start_player,
            claim_size=jnp.array(0, dtype=jnp.int32),
            claim_rank=jnp.array(0, dtype=jnp.int32),
            pending_play_hand=self._empty_hand(),
            pending_play_count=jnp.array(0, dtype=jnp.int32),
            has_current_rank=jnp.array(False),
            current_rank=jnp.array(0, dtype=jnp.int32),
            rank_choice_pending=jnp.array(True),
            challenge_status=jnp.array(3, dtype=jnp.int32),
            challenge_hand=self._empty_hand(),
            absorbing=jnp.zeros((self.num_agents,), dtype=bool),
            done=jnp.array(False),
            game_winner=jnp.zeros((self.num_agents,), dtype=bool),
            timestep=0,
        )
        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: BluffState, action: IntArray
    ) -> tuple[BluffState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]]:
        reward = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.int32)

        # phase 0: choose claim size (1..4)
        def do_claim() -> tuple[BluffState, FloatArray]:
            claim_size = jnp.minimum(action, jnp.int32(3)) + 1
            next_state = state.replace(
                phase=jnp.array(1, dtype=jnp.int32),
                claim_size=claim_size,
                pending_play_count=jnp.array(0, dtype=jnp.int32),
                pending_play_hand=self._empty_hand(),
                challenge_status=jnp.array(3, dtype=jnp.int32),
                challenge_hand=self._empty_hand(),
                challenge_target_idx=state.current_player_idx,
            )
            return next_state, reward

        # phase 1: recursively select play cards until claim size reached
        def do_play() -> tuple[BluffState, FloatArray]:
            chosen_rank = self._action_offset_to_rank(state, action)
            new_pending_play_hand = state.pending_play_hand.at[chosen_rank].add(1.0)
            new_pending_play_count = state.pending_play_count + 1

            first_played_rank = jnp.argmax(new_pending_play_hand > 0).astype(jnp.int32)
            keep_rank = state.has_current_rank & (~state.rank_choice_pending)
            new_claim_rank = lax.cond(
                keep_rank,
                lambda: state.current_rank,
                lambda: first_played_rank,
            )
            new_has_rank = jnp.array(True)
            new_current_rank = new_claim_rank

            finished = new_pending_play_count >= state.claim_size

            def finish_play() -> BluffState:
                next_hands = state.agent_hands.at[state.current_player_idx].add(
                    -new_pending_play_hand
                )
                next_hand_sizes = next_hands.sum(axis=1).astype(jnp.int32)
                claim_vec = jax.nn.one_hot(
                    new_claim_rank, self.num_ranks, dtype=jnp.float32
                ) * state.claim_size.astype(jnp.float32)
                next_pile_hand = state.pile_hand + new_pending_play_hand
                next_pile_claims = state.pile_claims + claim_vec
                next_pile_size = next_pile_hand.sum().astype(jnp.int32)
                first_challenger = self._next_player(state.current_player_idx)
                return state.replace(
                    agent_hands=next_hands,
                    agent_hand_sizes=next_hand_sizes,
                    pile_hand=next_pile_hand,
                    pile_claims=next_pile_claims,
                    pile_size=next_pile_size,
                    claim_rank=new_claim_rank,
                    current_rank=new_current_rank,
                    has_current_rank=new_has_rank,
                    phase=jnp.array(2, dtype=jnp.int32),
                    current_player_idx=first_challenger,
                    challenge_target_idx=state.current_player_idx,
                    pending_play_hand=new_pending_play_hand,
                    pending_play_count=new_pending_play_count,
                    challenge_status=jnp.array(3, dtype=jnp.int32),
                    challenge_hand=self._empty_hand(),
                    rank_choice_pending=jnp.array(False),
                )

            def continue_play() -> BluffState:
                return state.replace(
                    pending_play_hand=new_pending_play_hand,
                    pending_play_count=new_pending_play_count,
                    claim_rank=new_claim_rank,
                    has_current_rank=new_has_rank,
                    current_rank=new_current_rank,
                    rank_choice_pending=jnp.array(False),
                )

            next_state = lax.cond(finished, finish_play, continue_play)
            return next_state, reward

        # phase 2: query challengers in fixed turn order
        def do_challenge() -> tuple[BluffState, FloatArray]:
            target = state.challenge_target_idx
            challenger = state.current_player_idx
            challenge_called = action == 0
            claim_is_true = (
                state.pending_play_hand[state.claim_rank] == state.claim_size
            )

            def resolve_challenge() -> tuple[BluffState, FloatArray]:
                loser = jnp.where(claim_is_true, challenger, target)
                winner = jnp.where(claim_is_true, target, challenger)
                next_hands = state.agent_hands.at[loser].add(state.pile_hand)
                next_hand_sizes = next_hands.sum(axis=1).astype(jnp.int32)

                outcome_reward = jnp.zeros((self.num_agents,), dtype=jnp.float32)
                outcome_reward = outcome_reward.at[winner].add(self._reward_challenge)
                outcome_reward = outcome_reward.at[loser].add(
                    -state.pile_size.astype(jnp.float32) * self._reward_card
                )

                next_state = state.replace(
                    agent_hands=next_hands,
                    agent_hand_sizes=next_hand_sizes,
                    pile_hand=self._empty_hand(),
                    pile_claims=self._empty_hand(),
                    pile_size=jnp.array(0, dtype=jnp.int32),
                    phase=jnp.array(0, dtype=jnp.int32),
                    current_player_idx=winner.astype(jnp.int32),
                    round_start_player_idx=winner.astype(jnp.int32),
                    challenge_target_idx=winner.astype(jnp.int32),
                    pending_play_hand=self._empty_hand(),
                    pending_play_count=jnp.array(0, dtype=jnp.int32),
                    claim_size=jnp.array(0, dtype=jnp.int32),
                    challenge_status=jnp.where(
                        claim_is_true,
                        jnp.array(0, dtype=jnp.int32),
                        jnp.array(1, dtype=jnp.int32),
                    ),
                    challenge_hand=state.pending_play_hand,
                    has_current_rank=jnp.array(True),
                    # keep previous rank visible at round reset
                    current_rank=state.claim_rank,
                    rank_choice_pending=jnp.array(True),
                )
                return next_state, outcome_reward

            def pass_challenge() -> tuple[BluffState, FloatArray]:
                next_challenger = self._next_player(challenger)
                all_declined = next_challenger == target

                def finish_no_challenge() -> tuple[BluffState, FloatArray]:
                    next_player = self._next_player(target)
                    next_state = state.replace(
                        phase=jnp.array(0, dtype=jnp.int32),
                        current_player_idx=next_player,
                        challenge_target_idx=next_player,
                        pending_play_hand=self._empty_hand(),
                        pending_play_count=jnp.array(0, dtype=jnp.int32),
                        claim_size=jnp.array(0, dtype=jnp.int32),
                        challenge_status=jnp.array(2, dtype=jnp.int32),
                        challenge_hand=self._empty_hand(),
                        has_current_rank=jnp.array(True),
                        current_rank=(state.claim_rank + 1) % self.num_ranks,
                        rank_choice_pending=jnp.array(False),
                    )
                    no_challenge_reward = jnp.zeros(
                        (self.num_agents,), dtype=jnp.float32
                    )
                    no_challenge_reward = no_challenge_reward.at[target].add(
                        state.pending_play_hand.sum() * self._reward_card
                    )
                    return next_state, no_challenge_reward

                def continue_challenge() -> tuple[BluffState, FloatArray]:
                    next_state = state.replace(current_player_idx=next_challenger)
                    return next_state, reward

                return lax.cond(
                    all_declined,
                    finish_no_challenge,
                    continue_challenge,
                )

            return lax.cond(challenge_called, resolve_challenge, pass_challenge)

        next_state, phase_reward = lax.switch(
            state.phase, [do_claim, do_play, do_challenge]
        )
        reward = reward + phase_reward

        next_timestep = state.timestep + 1
        game_winner = next_state.agent_hand_sizes == 0
        game_over = game_winner.any() & (next_state.phase == 0)
        done = game_over | (next_timestep >= self.horizon)
        absorbing = jnp.broadcast_to(done, (self.num_agents,))

        reward = reward + jnp.where(game_winner, self._reward_win, 0.0)

        next_state = next_state.replace(
            timestep=next_timestep,
            game_winner=game_winner,
            done=done,
            absorbing=absorbing,
        )
        obs = self.obs_from_state(next_state)
        info = {
            "phase": next_state.phase,
            "claim_size": next_state.claim_size,
            "claim_rank": next_state.claim_rank,
            "challenge_status": next_state.challenge_status,
            "game_winner": next_state.game_winner,
        }
        return next_state, obs, reward, absorbing, done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.action_dim)
