import jax
import jax.numpy as jnp
from bluffjax import make, available_envs

# List available environments
print(available_envs())

# Create an environment
env = make("kuhn_poker")  # or "leduc_holdem", "goofspiel", "bluff", etc.

# With custom kwargs
env = make("bluff", num_agents=4, horizon=100)

# Reset and run random rollouts
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)


def random_rollout(key, env, state, obs, max_steps=100):
    def step(carry, _):
        key, state, obs = carry
        avail = env.get_avail_actions(state)
        logits = jnp.where(avail, 0.0, -1e9)
        key, k = jax.random.split(key)
        action = jax.random.categorical(k, logits)
        key, k2 = jax.random.split(key)
        state, obs, reward, absorbing, done, info = env.step(k2, state, action)
        return (key, state, obs), (state, obs, reward, done)

    (_, _, _), (states, observations, rewards, dones) = jax.lax.scan(
        step, (key, state, obs), None, length=max_steps
    )
    return states, observations, rewards, dones


# Run rollout
states, observations, rewards, dones = random_rollout(key, env, state, obs)

# Run rollout with JIT
jit_rollout = jax.jit(random_rollout, static_argnums=(1,))
states, observations, rewards, dones = jit_rollout(key, env, state, obs)
