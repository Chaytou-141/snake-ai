"""Validated, serialisable central configuration for CPU DQN experiments."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from snake_ai.rl import EnvConfig, RewardConfig


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.0005
    gamma: float = 0.99
    replay_capacity: int = 50_000
    batch_size: int = 64
    learning_starts: int = 1_000
    target_update_interval: int = 1_000  # Optimiser steps, not environment moves.
    gradient_clip_norm: float = 10.0
    hidden_size: int = 128
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 50_000  # Environment moves, not optimiser steps.
    episodes: int = 500
    max_environment_steps: int = 100_000  # Additional moves allowed in this run.
    checkpoint_interval_episodes: int = 100
    seed: int = 7
    training_seed_start: int = 100_000
    evaluation_seed_start: int = 900_000
    evaluation_episodes: int = 100
    random_policy_seed_start: int = 2_000_000
    cpu_threads: int = 1
    environment: EnvConfig = field(default_factory=EnvConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        positive = ("replay_capacity", "batch_size", "learning_starts", "target_update_interval",
                    "hidden_size", "epsilon_decay_steps", "episodes", "max_environment_steps",
                    "checkpoint_interval_episodes", "evaluation_episodes", "cpu_threads")
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "training_seed_start", "evaluation_seed_start", "random_policy_seed_start"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("gamma", "epsilon_start", "epsilon_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.epsilon_min > self.epsilon_start or self.batch_size > self.replay_capacity:
            raise ValueError("Require epsilon_min <= epsilon_start and batch_size <= replay_capacity")
        if self.training_seed_start + self.episodes > self.evaluation_seed_start:
            raise ValueError("Training seed range must be below reserved evaluation seeds")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        values = dict(data)
        values["environment"] = EnvConfig(**values.get("environment", {}))
        values["rewards"] = RewardConfig(**values.get("rewards", {}))
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
