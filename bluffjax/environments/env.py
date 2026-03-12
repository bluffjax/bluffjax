"""
Abstract class for multi-agent environments.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import abc
from functools import partial
from flax import struct
from bluffjax.utils.typing import (
    Any,
    Array,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)
from bluffjax.environments.spaces import Space


@struct.dataclass
class ParallelState:
    """
    Generic parallel game state.

    Order:
    1) private_info
    2) public_info
    3) start_player_idx
    4) current_player_idx
    5) returns
    6) absorbing
    7) done (all absorbing or global horizon reached)
    8) timestep
    """

    private_info: FloatArray
    public_info: FloatArray
    start_player_idx: int
    current_player_idx: int
    returns: FloatArray
    absorbing: BoolArray
    done: bool
    timestep: int


@struct.dataclass
class AECState:
    """
    Generic AEC (agent-environment-cycle) game state.

    Order:
    1) private_info
    2) public_info
    3) start_player_idx
    4) current_player_idx
    5) returns
    6) absorbing
    7) done (all absorbing or global horizon reached)
    8) timestep
    """

    private_info: FloatArray
    public_info: FloatArray
    start_player_idx: int
    current_player_idx: int
    returns: FloatArray
    absorbing: BoolArray
    done: bool
    timestep: int


class ParallelEnv(abc.ABC):
    """
    Abstract base class for all parallel BluffJAX Environments.

    All observations, actions, rewards, and absorbing states are provided for each agent at every step.
    """

    def __init__(self, num_agents: int, horizon: int) -> None:
        """
        Args:
            num_agents (int): maximum number of agents within the environment
            horizon (int): maximum number of steps for one episode
        """
        self.num_agents = num_agents
        self.agent_idxs = jnp.arange(num_agents)
        self.agents = [f"agent_{idx}" for idx in self.agent_idxs]
        self.horizon = horizon

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[ParallelState, FloatArray]:
        """
        Resets the environment.

        Args:
            rng (PRNGKeyArray): random number generator

        Returns:
            State (ParallelState): environment state
            Observations (FloatArray): observations for each agent
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: PRNGKeyArray,
        state: ParallelState,
        action: IntArray,
    ) -> tuple[
        ParallelState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Runs self.reset and self.step_env and selects the next state and next obs based on the done flag.

        Args:
            rng (PRNGKeyArray): random key
            state (ParallelState): environment state
            actions (IntArray): agent actions

        Returns:
            state_next (ParallelState): next environment state
            obs_next (FloatArray): next observations
            reward (FloatArray): rewards
            absorbing (BoolArray): absorbing state reached for each agent
            done (bool): absorbing state reached for all agents or episode truncated
            info (dict[str, Any]): info dictionary
        """
        rng_step, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.reset(rng_reset)
        state, obs, reward, absorbing, done, info = self.step_env(
            rng_step, state, action
        )
        (state, obs) = lax.cond(
            done, lambda: (state_reset, obs_reset), lambda: (state, obs)
        )
        return state, obs, reward, absorbing, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: ParallelState,
        action: IntArray,
    ) -> tuple[
        ParallelState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Performs step transitions in the environment.

        Args:
            rng (PRNGKeyArray): random key
            state (ParallelState): environment state
            actions (IntArray): agent actions

        Returns:
            state_next (ParallelState): next environment state
            obs_next (FloatArray): next observations
            reward (FloatArray): rewards
            absorbing (BoolArray): absorbing state reached for each agent
            done (bool): absorbing state reached for all agents or episode truncated
            info (dict[str, Any]): info dictionary
        """

        raise NotImplementedError

    def obs_from_state(self, state: ParallelState) -> FloatArray:
        """
        Gets observation from state

        Args:
            state (ParallelState): Environment state

        Returns:
            obs (FloatArray): observations

        Order:
        1) private_info
        2) public_info
        3) relative idx of current player
        4) auxiliary info (current round, new round, etc.)
        """
        raise NotImplementedError

    def observation_space(self) -> Space:
        """
        Observation spaces.

        Returns:
            space (Space): observation spaces
        """
        raise NotImplementedError

    def action_space(self) -> Space:
        """Action spaces.

        Returns:
            space (Space): action spaces
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: ParallelState) -> BoolArray:
        """Returns the available actions for each agent.

        Args:
            state (ParallelState): environment state

        Returns:
            available actions (BoolArray): available actions
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def _rel_array(self, arr: FloatArray) -> FloatArray:
        """
        Shifts array relative to each agent
        """
        return jax.vmap(lambda i: jnp.roll(arr, -i, axis=0))(
            jnp.arange(self.num_agents)
        ).astype(jnp.float32)


class AECEnv(abc.ABC):
    """
    Abstract base class for all agent-environment-cycle BluffJAX Environments.

    Observations, actions, and absorbing states are provided for the current agent at every step.
    Rewards are provided for all agents at every step.
    """

    def __init__(self, num_agents: int, horizon: int) -> None:
        """
        Args:
            num_agents (int): maximum number of agents within the environment
            horizon (int): maximum number of steps for one episode
        """
        self.num_agents = num_agents
        self.agent_idxs = jnp.arange(num_agents)
        self.agents = [f"agent_{idx}" for idx in self.agent_idxs]
        self.horizon = horizon

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[AECState, FloatArray]:
        """
        Resets the environment.

        Args:
            rng (PRNGKeyArray): random number generator

        Returns:
            State (AECState): environment state
            Observations (FloatArray): observations for each agent
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: PRNGKeyArray,
        state: ParallelState,
        action: IntArray,
    ) -> tuple[
        ParallelState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Performs step transitions in the environment.

        Args:
            rng (PRNGKeyArray): random key
            state (ParallelState): environment state
            actions (IntArray): agent actions

        Returns:
            state_next (ParallelState): next environment state
            obs_next (FloatArray): next observations
            reward (FloatArray): rewards
            absorbing (BoolArray): absorbing state reached
            done (bool): absorbing state reached or episode truncated
            info (dict[str, Any]): info dictionary
        """

        rng_step, rng_reset = jax.random.split(rng)
        state_reset, obs_reset = self.reset(rng_reset)
        state_next, obs_next, reward, absorbing, done, info = self.step_env(
            rng_step, state, action
        )
        state_final = lax.cond(done, lambda: state_reset, lambda: state_next)
        obs_final = lax.cond(done, lambda: obs_reset, lambda: obs_next)

        return state_final, obs_final, reward, absorbing, done, info

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: AECState,
        action: IntArray,
    ) -> tuple[AECState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]]:
        """Performs step transitions in the environment."""
        raise NotImplementedError

    def obs_from_state(self, state: AECState) -> FloatArray:
        """
        Gets observation from state

        Args:
            state (AECState): Environment state

        Returns:
            obs (FloatArray): observations

        Order:
        1) private_info
        2) public_info
        3) current_player_idx
        4) auxiliary info (current round, new round, etc.)
        """
        raise NotImplementedError

    def observation_space(self) -> Space:
        """
        Observation spaces.

        Returns:
            space (Space): observation spaces
        """
        raise NotImplementedError

    def action_space(self) -> Space:
        """Action spaces.

        Returns:
            space (Space): action spaces
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: AECState) -> BoolArray:
        """Returns the available actions for each agent.

        Args:
            state (AECState): environment state

        Returns:
            available actions (BoolArray): available actions
        """
        raise NotImplementedError
