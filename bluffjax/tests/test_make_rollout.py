"""Smoke test: create each environment via make() and run random rollouts."""

import jax
import jax.numpy as jnp
import pytest

from bluffjax import available_envs, make


def _random_rollout(key, env, max_steps: int = 50):
    """Run a short random rollout."""
    state, obs = env.reset(key)

    def step(carry, _):
        key, state, obs = carry
        avail = env.get_avail_actions(state)
        logits = jnp.where(avail, 0.0, -1e9)
        key, k = jax.random.split(key)
        action = jax.random.categorical(k, logits)
        key, k2 = jax.random.split(key)
        state, obs, reward, absorbing, done, info = env.step(k2, state, action)
        return (key, state, obs), (state, obs, reward, done)

    (_, _, _), (states, obs, rewards, dones) = jax.lax.scan(
        step, (key, state, obs), None, length=max_steps
    )
    return states, obs, rewards, dones


@pytest.mark.parametrize("env_id", available_envs())
def test_make_and_rollout(env_id: str) -> None:
    """Each registered env can be created and run for a few steps."""
    env = make(env_id)
    key = jax.random.PRNGKey(0)
    states, obs, rewards, dones = _random_rollout(key, env)
    assert states is not None
    assert obs is not None
    assert rewards is not None
    assert dones is not None


def test_available_envs() -> None:
    """available_envs returns non-empty tuple."""
    envs = available_envs()
    assert isinstance(envs, tuple)
    assert len(envs) > 0
    assert "kuhn_poker" in envs


def test_make_invalid_env_raises() -> None:
    """make() with invalid env_id raises ValueError."""
    with pytest.raises(ValueError, match="not in registered"):
        make("invalid_env_id")


def test_make_with_kwargs() -> None:
    """make() forwards kwargs to constructor."""
    env = make("leduc_holdem", num_agents=2, horizon=20)
    assert env.num_agents == 2
    assert env.horizon == 20
