"""Bounded replay memory with copied observations and an independent RNG."""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Iterable

import torch


def observation_values(values: Iterable[float], dimension: int = 11) -> tuple[float, ...]:
    values = tuple(values)
    if len(values) != dimension:
        raise ValueError(f"Expected {dimension} observation values")
    if any(not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError("Observation values must be finite numbers")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class Experience:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int | None = None,
                 observation_dim: int = 11) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(observation_dim) is not int or observation_dim < 1:
            raise ValueError("observation_dim must be a positive integer")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self._memory: deque[Experience] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def add(self, state: Iterable[float], action: int, reward: float,
            next_state: Iterable[float], terminated: bool, truncated: bool) -> None:
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("action must be integer 0, 1, or 2")
        if not isinstance(reward, Real) or isinstance(reward, bool) or not math.isfinite(reward):
            raise ValueError("reward must be a finite number")
        if type(terminated) is not bool or type(truncated) is not bool or (terminated and truncated):
            raise ValueError("termination flags must be booleans and cannot both be true")
        self._memory.append(Experience(observation_values(state, self.observation_dim),
            action, float(reward), observation_values(next_state, self.observation_dim),
            terminated, truncated))

    def sample(self, batch_size: int) -> ReplayBatch:
        if type(batch_size) is not int or not 1 <= batch_size <= len(self):
            raise ValueError("batch_size must be positive and no greater than available experiences")
        samples = self._rng.sample(list(self._memory), batch_size)
        return ReplayBatch(
            torch.tensor([row.state for row in samples], dtype=torch.float32),
            torch.tensor([row.action for row in samples], dtype=torch.int64),
            torch.tensor([row.reward for row in samples], dtype=torch.float32),
            torch.tensor([row.next_state for row in samples], dtype=torch.float32),
            torch.tensor([row.terminated for row in samples], dtype=torch.bool),
            torch.tensor([row.truncated for row in samples], dtype=torch.bool),
        )
