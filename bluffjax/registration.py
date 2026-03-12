"""
Environment registration for BluffJAX.

Provides a simple way to build each environment with kwargs.
"""

from typing import Union

from bluffjax.environments.env import AECEnv, ParallelEnv

registered_envs: list[str] = [
    "kuhn_poker",
    "leduc_holdem",
    "goofspiel",
    "kemps",
    "bluff",
    "five_card_draw",
    "seven_card_stud",
    "texas_limit_holdem",
    "texas_nolimit_holdem",
    "werewolf",
]


def make(env_id: str, **env_kwargs) -> Union[AECEnv, ParallelEnv]:
    """
    Create an environment by ID with optional kwargs.

    Args:
        env_id: Environment identifier (e.g. "kuhn_poker", "leduc_holdem")
        **env_kwargs: Keyword arguments forwarded to the environment constructor

    Returns:
        Environment instance (AECEnv or ParallelEnv)

    Raises:
        ValueError: If env_id is not in registered_envs
    """
    if env_id not in registered_envs:
        raise ValueError(
            f"{env_id} is not in registered bluffjax environments. "
            f"Available: {registered_envs}"
        )

    if env_id == "kuhn_poker":
        from bluffjax.environments.kuhn_poker.kuhn_poker import KuhnPoker

        return KuhnPoker(**env_kwargs)
    elif env_id == "leduc_holdem":
        from bluffjax.environments.leduc_holdem.leduc_holdem import LeducHoldem

        return LeducHoldem(**env_kwargs)
    elif env_id == "goofspiel":
        from bluffjax.environments.goofspiel.goofspiel import Goofspiel

        return Goofspiel(**env_kwargs)
    elif env_id == "kemps":
        from bluffjax.environments.kemps.kemps import Kemps

        return Kemps(**env_kwargs)
    elif env_id == "bluff":
        from bluffjax.environments.bluff.bluff import Bluff

        return Bluff(**env_kwargs)
    elif env_id == "five_card_draw":
        from bluffjax.environments.five_card_draw.five_card_draw import FiveCardDraw

        return FiveCardDraw(**env_kwargs)
    elif env_id == "seven_card_stud":
        from bluffjax.environments.seven_card_stud.seven_card_stud import (
            SevenCardStud,
        )

        return SevenCardStud(**env_kwargs)
    elif env_id == "texas_limit_holdem":
        from bluffjax.environments.texas_limit_holdem.texas_limit_holdem import (
            TexasLimitHoldem,
        )

        return TexasLimitHoldem(**env_kwargs)
    elif env_id == "texas_nolimit_holdem":
        from bluffjax.environments.texas_nolimit_holdem.texas_nolimit_holdem import (
            TexasNoLimitHoldem,
        )

        return TexasNoLimitHoldem(**env_kwargs)
    elif env_id == "werewolf":
        from bluffjax.environments.werewolf.werewolf import Werewolf

        return Werewolf(**env_kwargs)
    else:
        raise ValueError(f"Invalid environment: {env_id}")


def available_envs() -> tuple[str, ...]:
    """Return tuple of valid environment IDs."""
    return tuple(registered_envs)
