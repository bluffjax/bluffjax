import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import numpy as np
import jax.numpy as jnp
from bluffjax.utils.typing import (
    FloatArray,
)


class ActorCriticDiscreteMLP(nn.Module):
    action_dim: int
    hidden_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x) -> tuple[FloatArray, FloatArray]:
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "tanh":
            activation = nn.tanh

        # actor
        actor = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor = activation(actor)
        actor = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor)
        actor = activation(actor)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)

        # critic
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)

        return (
            actor_logits,
            jnp.squeeze(critic, axis=-1),
        )


class CriticMLP(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x) -> FloatArray:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorDiscreteMLP(nn.Module):
    action_dim: int
    hidden_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x) -> FloatArray:
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "tanh":
            activation = nn.tanh

        # actor
        actor = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor = activation(actor)
        actor = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor)
        actor = activation(actor)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)
        return actor_logits


class QNetworkDiscreteMLP(nn.Module):
    """PQN-style Q-network with LayerNorm after each hidden layer."""

    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x) -> FloatArray:
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        return x
