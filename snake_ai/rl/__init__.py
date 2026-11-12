"""Lightweight headless RL interface; no ML or rendering dependencies."""

from .environment import FEATURE_NAMES, EnvConfig, RewardConfig, SnakeEnv, relative_direction
from .random_agent import RandomAgent

__all__ = ["FEATURE_NAMES", "EnvConfig", "RewardConfig", "SnakeEnv",
           "relative_direction", "RandomAgent"]
