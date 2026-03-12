"""
Spaces for BluffJAX environments.
"""

import jax
import jax.numpy as jnp
from bluffjax.utils.typing import (
    Array,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)


class Space(object):
    """
    Minimal jittable class for abstract spaces.
    """

    def sample(self, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError

    def contains(self, x: Array) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """
    Class for discrete spaces.
    """

    def __init__(self, num_categories: int) -> None:
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()

    def sample(self, rng: PRNGKeyArray) -> Array:
        return jax.random.randint(rng, shape=self.shape, minval=0, maxval=self.n)

    def contains(self, x: Array) -> bool:
        return jnp.logical_and(x >= 0, x < self.n).astype(bool)
