"""
BluffJAX: JAX-based multi-agent imperfect-information game environments.

Provides poker, bluffing, and social deduction games for reinforcement learning research.
"""

from bluffjax.registration import available_envs, make, registered_envs

__version__ = "1.0.0"
__all__ = ["make", "registered_envs", "available_envs"]
