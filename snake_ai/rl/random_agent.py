"""A uniform random policy that never learns or inspects observations."""

import random


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Explicit seed restarts action sequence; None preserves its stream."""
        if seed is not None:
            self._rng.seed(seed)

    def act(self) -> int:
        return self._rng.randrange(3)
