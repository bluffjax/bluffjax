from bluffjax.utils.typing import PyTree, Array

import jax
import jax.numpy as jnp


def pytree_norm(pytree: PyTree) -> Array:
    """
    Computes the L2 norm of a pytree
    """
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), pytree)
    total_square = jax.tree.reduce(lambda leaf_1, leaf_2: leaf_1 + leaf_2, squares)
    return jnp.sqrt(total_square)


jprint = lambda *args: [jax.debug.print("{var}", var=arg) for arg in args]
