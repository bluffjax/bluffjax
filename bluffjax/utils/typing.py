from typing import Any
from jaxtyping import Array, Float, Int, Bool, PyTree, PRNGKeyArray

# for type annotations only
Any = Any
Array = Array
FloatArray = Float[Array, "..."]
IntArray = Int[Array, "..."]
BoolArray = Bool[Array, "..."]
PyTree = PyTree
PRNGKeyArray = PRNGKeyArray
