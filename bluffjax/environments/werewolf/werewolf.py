"""
Werewolf (Mafia) social deduction environment.

Roles: werewolves (2), doctor (1), seer (1), villagers (2).
Phases: night (doctor -> seer -> werewolves), accuse, vote.
All observations and actions are relative to the current player.
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

# Role constants
VILLAGER = 0
WEREWOLF = 1
DOCTOR = 2
SEER = 3

# Phase constants
PHASE_NIGHT = 0
PHASE_ACCUSE = 1
PHASE_VOTE = 2

# Night subphase: who acts (doctor, seer, werewolf0, werewolf1)
NIGHT_DOCTOR = 0
NIGHT_SEER = 1
NIGHT_WEREWOLF_0 = 2
NIGHT_WEREWOLF_1 = 3


@struct.dataclass
class WerewolfState:
    """Werewolf game state."""

    roles: IntArray  # (num_agents,) 0=villager, 1=werewolf, 2=doctor, 3=seer
    alive: BoolArray  # (num_agents,)
    phase: IntArray  # 0=night, 1=accuse, 2=vote
    night_subphase: IntArray  # 0=doctor, 1=seer, 2=ww0, 3=ww1
    current_player_idx: IntArray

    # Night order: [doctor_idx, seer_idx, werewolf0_idx, werewolf1_idx]
    night_order: IntArray  # (4,)

    # Night actions (absolute target indices, -1 = not yet chosen)
    doctor_target: IntArray
    seer_target: IntArray
    werewolf_targets: IntArray  # (2,) - each werewolf's target

    # Seer's investigation results: (num_agents,) -1=villager, 0=unchecked, 1=werewolf
    seer_results: FloatArray

    # Accuse phase: (num_agents,) who each player accused (-1 = no accusation)
    accusations: IntArray

    # Vote phase: (num_agents,) who each player voted for
    votes: IntArray

    # Phase progress: for accuse/vote, how many have acted
    phase_progress: IntArray

    absorbing: BoolArray
    done: bool
    game_winner: IntArray  # 0=humans, 1=werewolves
    timestep: int


class Werewolf(AECEnv):
    """Werewolf AEC environment with doctor and seer roles."""

    def __init__(
        self,
        num_agents: int = 6,
        num_werewolves: int = 2,
        horizon: int = 200,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.num_werewolves = num_werewolves
        self.num_roles = 4  # villager, werewolf, doctor, seer

        # Action: 0 to num_agents-1 = relative target, num_agents = noop (accuse only)
        self.num_actions = num_agents + 1

        # Obs dim: role(4) + phase(3) + alive(num_agents) + werewolf_teammates(num_agents)
        # + seer_results(num_agents*3 for ternary) + accusations(num_agents)
        self.obs_dim = 4 + 3 + num_agents + num_agents + num_agents * 3 + num_agents

    def _to_relative(
        self, absolute_idx: IntArray, current_player: IntArray
    ) -> IntArray:
        """Convert absolute index to relative (0 = self)."""
        return (absolute_idx - current_player) % self.num_agents

    def _to_absolute(
        self, relative_idx: IntArray, current_player: IntArray
    ) -> IntArray:
        """Convert relative index to absolute."""
        return (current_player + relative_idx) % self.num_agents

    def _roll_for_perspective(
        self, arr: FloatArray, current_player: IntArray
    ) -> FloatArray:
        """Roll array so current player is at index 0 (relative perspective)."""
        return jnp.roll(arr, -current_player, axis=0)

    def _get_next_alive_player(self, current: IntArray, alive: BoolArray) -> IntArray:
        """Get next alive player starting from (current + 1) % n."""

        def scan_fn(carry, offset):
            idx = (current + 1 + offset) % self.num_agents
            is_alive = alive[idx]
            found_idx, found = carry
            new_found = jnp.where(~found & is_alive, idx, found_idx)
            new_found_flag = found | is_alive
            return (new_found, new_found_flag), None

        (result, _), _ = lax.scan(
            scan_fn, (current, False), jnp.arange(self.num_agents)
        )
        return result

    def _get_first_alive_from(self, start: IntArray, alive: BoolArray) -> IntArray:
        """Get first alive player at or after start."""

        def scan_fn(carry, offset):
            idx = (start + offset) % self.num_agents
            is_alive = alive[idx]
            found_idx, found = carry
            new_found = jnp.where(~found & is_alive, idx, found_idx)
            new_found_flag = found | is_alive
            return (new_found, new_found_flag), None

        (result, _), _ = lax.scan(scan_fn, (start, False), jnp.arange(self.num_agents))
        return result

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: WerewolfState) -> FloatArray:
        """Observation for current player, all relative."""
        cp = state.current_player_idx

        # 1. One-hot role (4 dims)
        role_oh = jax.nn.one_hot(state.roles[cp], self.num_roles, dtype=jnp.float32)

        # 2. One-hot phase (3 dims)
        phase_oh = jax.nn.one_hot(state.phase, 3, dtype=jnp.float32)

        # 3. Binary living players, relative (num_agents dims)
        alive_relative = self._roll_for_perspective(state.alive.astype(jnp.float32), cp)

        # 4. Werewolf teammates relative (num_agents dims) - only non-zero if current is werewolf
        is_werewolf = state.roles[cp] == WEREWOLF
        teammate_mask = (state.roles == WEREWOLF) & (self.agent_idxs != cp)
        teammate_relative = self._roll_for_perspective(
            teammate_mask.astype(jnp.float32), cp
        )
        werewolf_teammates = jnp.where(
            is_werewolf, teammate_relative, jnp.zeros(self.num_agents)
        )

        # 5. Seer results: ternary per player (-1, 0, 1), relative. Encode as 3 values each.
        seer_rel = self._roll_for_perspective(state.seer_results, cp)
        is_seer = state.roles[cp] == SEER

        # Encode each of num_agents positions: 3 dims for -1/0/1
        def encode_ternary(val):
            return jax.nn.one_hot((val + 1).astype(jnp.int32), 3, dtype=jnp.float32)

        seer_encoded = jax.vmap(encode_ternary)(seer_rel).reshape(-1)
        seer_obs = jnp.where(is_seer, seer_encoded, jnp.zeros(self.num_agents * 3))

        # 6. Accusations: binary relative - who was accused
        # accused_by_any[j] = 1 if any player accused j
        valid_acc = state.accusations >= 0
        accused_by_any = jnp.any(
            (state.accusations[None, :] == jnp.arange(self.num_agents)[:, None])
            & valid_acc[None, :],
            axis=1,
        ).astype(jnp.float32)
        accusations_relative = self._roll_for_perspective(accused_by_any, cp)

        obs = jnp.concatenate(
            [
                role_oh,
                phase_oh,
                alive_relative,
                werewolf_teammates,
                seer_obs,
                accusations_relative,
            ]
        )
        return obs.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WerewolfState) -> BoolArray:
        """Available actions for current player. Action i = relative target i, num_agents = noop."""
        cp = state.current_player_idx

        def night_avail():
            actor_idx = state.night_order[state.night_subphase]
            is_actor = (cp == actor_idx) & state.alive[actor_idx]
            doctor_mask = state.alive
            seer_mask = state.alive
            werewolf_mask = state.alive & (state.roles != WEREWOLF)
            night_mask = jnp.where(
                state.night_subphase == NIGHT_DOCTOR,
                doctor_mask,
                jnp.where(
                    state.night_subphase == NIGHT_SEER,
                    seer_mask,
                    werewolf_mask,
                ),
            )
            avail = jnp.concatenate(
                [
                    jnp.where(
                        is_actor, night_mask, jnp.zeros(self.num_agents, dtype=bool)
                    ),
                    jnp.array([~is_actor], dtype=bool),
                ]
            )
            return avail

        def accuse_avail():
            abs_indices = (cp + jnp.arange(self.num_agents)) % self.num_agents
            can_accuse = state.alive[abs_indices] & (jnp.arange(self.num_agents) != 0)
            avail = jnp.concatenate(
                [
                    jnp.where(
                        state.alive[cp],
                        can_accuse,
                        jnp.zeros(self.num_agents, dtype=bool),
                    ),
                    jnp.array([True], dtype=bool),
                ]
            )
            return avail

        def vote_avail():
            abs_indices = (cp + jnp.arange(self.num_agents)) % self.num_agents
            can_vote = state.alive[abs_indices] & (jnp.arange(self.num_agents) != 0)
            avail = jnp.concatenate(
                [
                    jnp.where(
                        state.alive[cp],
                        can_vote,
                        jnp.zeros(self.num_agents, dtype=bool),
                    ),
                    jnp.array([~state.alive[cp]], dtype=bool),
                ]
            )
            return avail

        return lax.switch(
            state.phase,
            [night_avail, accuse_avail, vote_avail],
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[WerewolfState, FloatArray]:
        """Initialize game with random role assignment."""
        rng_roles, rng_order = jax.random.split(rng)

        # Roles: 2 werewolves, 1 doctor, 1 seer, 2 villagers
        roles_template = jnp.array(
            [VILLAGER, VILLAGER, WEREWOLF, WEREWOLF, DOCTOR, SEER],
            dtype=jnp.int32,
        )
        roles = jax.random.permutation(rng_roles, roles_template)

        alive = jnp.ones(self.num_agents, dtype=bool)
        night_order = jnp.zeros(4, dtype=jnp.int32)
        doctor_idx = jnp.argmax(roles == DOCTOR)
        seer_idx = jnp.argmax(roles == SEER)
        ww_mask = roles == WEREWOLF
        ww_indices = jnp.where(ww_mask, self.agent_idxs, self.num_agents)
        ww0 = jnp.min(ww_indices)
        ww1 = jnp.min(jnp.where(ww_indices != ww0, ww_indices, self.num_agents + 1))
        night_order = night_order.at[0].set(doctor_idx)
        night_order = night_order.at[1].set(seer_idx)
        night_order = night_order.at[2].set(ww0)
        night_order = night_order.at[3].set(ww1)

        state = WerewolfState(
            roles=roles,
            alive=alive,
            phase=jnp.int32(PHASE_NIGHT),
            night_subphase=jnp.int32(0),
            current_player_idx=night_order[0],
            night_order=night_order,
            doctor_target=jnp.int32(-1),
            seer_target=jnp.int32(-1),
            werewolf_targets=jnp.array([-1, -1], dtype=jnp.int32),
            seer_results=jnp.zeros(self.num_agents, dtype=jnp.float32),
            accusations=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            votes=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=False,
            game_winner=jnp.array(-1, dtype=jnp.int32),
            timestep=0,
        )

        # Current player might be dead (shouldn't happen at start) - advance to first alive in night order
        first_actor = night_order[0]
        current = jnp.where(
            state.alive[first_actor],
            first_actor,
            self._get_first_alive_from(0, state.alive),
        )
        state = state.replace(current_player_idx=current)

        obs = self.obs_from_state(state)
        return state, obs

    def _resolve_night(
        self, state: WerewolfState, rng: PRNGKeyArray
    ) -> tuple[WerewolfState, PRNGKeyArray]:
        """Resolve night: apply kill (unless healed), update seer results."""
        rng_tie, rng_next = jax.random.split(rng)

        # Werewolf target: majority vote, random if tied
        ww0_tgt = state.werewolf_targets[0]
        ww1_tgt = state.werewolf_targets[1]
        # If only one werewolf alive, use their vote
        ww0_alive = state.alive[state.night_order[2]]
        ww1_alive = state.alive[state.night_order[3]]
        vote0 = jnp.where(ww0_alive, ww0_tgt, -1)
        vote1 = jnp.where(ww1_alive, ww1_tgt, -1)

        # Pick target: if same, use it; if different, random
        same_vote = (vote0 == vote1) & (vote0 >= 0)
        kill_target = jnp.where(
            same_vote,
            vote0,
            jnp.where(
                vote0 < 0,
                vote1,
                jnp.where(
                    vote1 < 0,
                    vote0,
                    jax.random.choice(
                        rng_tie,
                        jnp.array([vote0, vote1]),
                        shape=(),
                    ),
                ),
            ),
        )

        # Doctor protection
        healed = (state.doctor_target >= 0) & (kill_target == state.doctor_target)
        actual_kill = jnp.where(healed, -1, kill_target)

        # Apply death
        new_alive = jnp.where(
            (self.agent_idxs == actual_kill) & (actual_kill >= 0),
            False,
            state.alive,
        )

        # Seer results already updated when seer acted in step_env

        # Check win: werewolves win if ww >= humans
        num_ww = jnp.sum((state.roles == WEREWOLF) & new_alive)
        num_humans = jnp.sum((state.roles != WEREWOLF) & new_alive)
        werewolves_win = num_ww >= num_humans

        next_state = state.replace(
            alive=new_alive,
            phase=jnp.int32(PHASE_ACCUSE),
            current_player_idx=jnp.int32(0),
            accusations=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
            absorbing=jnp.broadcast_to(werewolves_win, (self.num_agents,)),
            done=werewolves_win,
            game_winner=jnp.where(werewolves_win, 1, -1),
        )
        return next_state, rng_next

    def _resolve_vote(
        self, state: WerewolfState, rng: PRNGKeyArray
    ) -> tuple[WerewolfState, PRNGKeyArray]:
        """Resolve vote: eliminate player with most votes, random if tied."""
        rng_tie, rng_next = jax.random.split(rng)

        # Count votes per player (only from alive voters)
        vote_counts = jnp.zeros(self.num_agents, dtype=jnp.float32)
        for i in range(self.num_agents):
            v = state.votes[i]
            vote_counts = jnp.where(
                state.alive[i] & (v >= 0),
                vote_counts.at[v].add(1.0),
                vote_counts,
            )

        # Break ties with random noise
        noise = jax.random.uniform(rng_tie, (self.num_agents,)) * 0.1
        vote_counts_noisy = vote_counts + noise
        elim_idx = jnp.argmax(vote_counts_noisy)
        max_votes = jnp.max(vote_counts)

        new_alive = jnp.where(
            (self.agent_idxs == elim_idx) & (max_votes > 0),
            False,
            state.alive,
        )

        # Check win: humans win if no werewolves left
        num_ww = jnp.sum((state.roles == WEREWOLF) & new_alive)
        humans_win = num_ww == 0
        num_humans = jnp.sum((state.roles != WEREWOLF) & new_alive)
        werewolves_win = num_ww >= num_humans

        game_over = humans_win | werewolves_win
        winner = jnp.where(humans_win, 0, jnp.where(werewolves_win, 1, -1))

        next_state = state.replace(
            alive=new_alive,
            phase=jnp.int32(PHASE_NIGHT),
            night_subphase=jnp.int32(0),
            current_player_idx=state.night_order[0],
            doctor_target=jnp.int32(-1),
            seer_target=jnp.int32(-1),
            werewolf_targets=jnp.array([-1, -1], dtype=jnp.int32),
            votes=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
            absorbing=jnp.broadcast_to(game_over, (self.num_agents,)),
            done=game_over,
            game_winner=winner,
        )
        return next_state, rng_next

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: WerewolfState, action: IntArray
    ) -> tuple[
        WerewolfState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Execute one step."""
        cp = state.current_player_idx

        # Convert relative action to absolute (for action < num_agents)
        abs_target = self._to_absolute(jnp.minimum(action, self.num_agents - 1), cp)
        is_noop = action >= self.num_agents

        rewards = jnp.zeros(self.num_agents, dtype=jnp.float32)

        def do_night():
            actor_idx = state.night_order[state.night_subphase]
            is_actor = (cp == actor_idx) & state.alive[actor_idx]

            def doctor_step():
                return state.replace(
                    doctor_target=jnp.where(is_noop, -1, abs_target),
                    night_subphase=jnp.int32(NIGHT_SEER),
                    current_player_idx=state.night_order[NIGHT_SEER],
                )

            def seer_step():
                new_seer = jnp.where(
                    (abs_target >= 0) & (self.agent_idxs == abs_target),
                    (state.roles == WEREWOLF).astype(jnp.float32) * 2 - 1,
                    state.seer_results,
                )
                return state.replace(
                    seer_target=jnp.where(is_noop, -1, abs_target),
                    seer_results=new_seer,
                    night_subphase=jnp.int32(NIGHT_WEREWOLF_0),
                    current_player_idx=state.night_order[NIGHT_WEREWOLF_0],
                )

            def werewolf_step():
                ww_idx = state.night_subphase - NIGHT_WEREWOLF_0
                new_targets = state.werewolf_targets.at[ww_idx].set(
                    jnp.where(is_noop, -1, abs_target)
                )
                next_sub = state.night_subphase + 1
                return state.replace(
                    werewolf_targets=new_targets,
                    night_subphase=next_sub,
                    current_player_idx=jnp.where(
                        next_sub < 4,
                        state.night_order[next_sub],
                        cp,
                    ),
                )

            def skip_action():
                next_sub = state.night_subphase + 1
                next_actor = jnp.where(
                    next_sub < 4,
                    state.night_order[next_sub],
                    state.night_order[0],
                )
                # When next_sub >= 4, set night_subphase=4 to trigger resolve
                return state.replace(
                    night_subphase=jnp.minimum(next_sub, 4),
                    current_player_idx=next_actor,
                )

            ns = lax.cond(
                is_actor,
                lambda: lax.switch(
                    jnp.clip(state.night_subphase, 0, 2),
                    [doctor_step, seer_step, werewolf_step],
                ),
                skip_action,
            )
            ns, rng_out = lax.cond(
                ns.night_subphase >= 4,
                lambda s, r: self._resolve_night(s, r),
                lambda s, r: (s, r),
                ns,
                rng,
            )
            ns = lax.cond(
                ns.phase == PHASE_ACCUSE,
                lambda s: s.replace(
                    current_player_idx=self._get_first_alive_from(0, s.alive),
                ),
                lambda s: s,
                ns,
            )
            return ns, rng_out

        def do_accuse():
            new_accusations = state.accusations.at[cp].set(
                jnp.where(is_noop, -1, abs_target)
            )
            # Iterate 0,1,2,...,num_agents-1 (each player acts once)
            next_idx = cp + 1
            all_done = next_idx >= self.num_agents
            next_player = jnp.where(all_done, 0, next_idx)
            ns = state.replace(
                accusations=new_accusations,
                current_player_idx=next_player,
            )
            ns = lax.cond(
                all_done,
                lambda s: s.replace(
                    phase=jnp.int32(PHASE_VOTE),
                    current_player_idx=jnp.int32(0),
                    votes=jnp.full(self.num_agents, -1, dtype=jnp.int32),
                ),
                lambda s: s,
                ns,
            )
            return ns, rng

        def do_vote():
            new_votes = state.votes.at[cp].set(jnp.where(is_noop, -1, abs_target))
            next_idx = cp + 1
            all_done = next_idx >= self.num_agents
            next_player = jnp.where(all_done, 0, next_idx)
            ns = state.replace(
                votes=new_votes,
                current_player_idx=next_player,
            )
            ns, rng_out = lax.cond(
                all_done,
                lambda s, r: self._resolve_vote(s, r),
                lambda s, r: (s, r),
                ns,
                rng,
            )
            return ns, rng_out

        next_state, rng = lax.switch(
            state.phase,
            [do_night, do_accuse, do_vote],
        )

        # Compute rewards at episode end
        def compute_rewards():
            is_ww = state.roles == WEREWOLF
            humans_win = next_state.game_winner == 0
            werewolves_win = next_state.game_winner == 1
            win = jnp.where(is_ww, werewolves_win, humans_win)
            return jnp.where(win, 10.0, -10.0).astype(jnp.float32)

        rewards = lax.cond(
            next_state.done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents, dtype=jnp.float32),
        )

        next_timestep = state.timestep + 1
        horizon_done = next_timestep >= self.horizon
        next_state = next_state.replace(timestep=next_timestep)
        next_state = lax.cond(
            horizon_done,
            lambda: next_state.replace(
                done=True,
                absorbing=jnp.ones(self.num_agents, dtype=bool),
                game_winner=jnp.int32(0),  # humans win on timeout?
            ),
            lambda: next_state,
        )

        obs = self.obs_from_state(next_state)
        absorbing = jnp.broadcast_to(next_state.done, (self.num_agents,))
        human_win = (next_state.game_winner == 0).astype(jnp.float32)
        werewolf_win = (next_state.game_winner == 1).astype(jnp.float32)
        is_werewolf = (next_state.roles == WEREWOLF).astype(jnp.float32)
        game_winner = is_werewolf * werewolf_win + (1.0 - is_werewolf) * human_win
        info = {
            "returns": rewards,
            "timestep": next_state.timestep,
            "game_winner": game_winner,
        }
        return next_state, obs, rewards, absorbing, next_state.done, info

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: PRNGKeyArray,
        state: WerewolfState,
        action: IntArray,
    ) -> tuple[
        WerewolfState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """AEC step with reset on done."""
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
        return Discrete(self.num_actions)

    def avail_actions(self, state: WerewolfState) -> BoolArray:
        """Alias for get_avail_actions for AECEnv compatibility."""
        return self.get_avail_actions(state)
