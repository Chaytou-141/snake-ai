"""Grid-based Snake rules. No display, clock, input devices, or Pygame needed."""

from enum import Enum
import random

Position = tuple[int, int]


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def is_opposite(self, other: "Direction") -> bool:
        dx, dy = self.value
        return other.value == (-dx, -dy)


class GameStatus(Enum):
    RUNNING = "running"
    LOST = "lost"
    WON = "won"


class SnakeGame:
    """One instance owns one board and RNG; snake positions are head first.

    Coordinates are (column, row), with (0, 0) at the top left. Call step()
    once per move. State attributes are readable by clients; normal gameplay
    changes them only through step() and reset().
    """

    def __init__(self, width: int = 20, height: int = 20,
                 seed: int | None = None) -> None:
        if (not isinstance(width, int) or isinstance(width, bool)
                or not isinstance(height, int) or isinstance(height, bool)
                or width < 2 or height < 2):
            raise ValueError("Board width and height must be integers of at least 2")
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.snake: list[Position] = []
        self.food: Position | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Restart; an explicit seed restarts the RNG sequence too.

        Without a seed, retain the current RNG stream so interactive restarts
        get fresh food while remaining reproducible across identical runs.
        """
        if seed is not None:
            self._rng.seed(seed)
        head_x, head_y = self.width // 2, self.height // 2
        length = min(3, head_x + 1)
        self.snake = [(head_x - offset, head_y) for offset in range(length)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.status = GameStatus.RUNNING
        self.food = self._spawn_food()

    def _spawn_food(self) -> Position | None:
        """Choose uniformly from empty cells, with stable row-major ordering."""
        occupied = set(self.snake)
        empty = [(x, y) for y in range(self.height) for x in range(self.width)
                 if (x, y) not in occupied]
        return self._rng.choice(empty) if empty else None

    def step(self, action: Direction | None = None) -> GameStatus:
        """Advance one cell and return status; terminal games stay unchanged.

        None continues forward. Reverse actions are ignored. A collision marks
        the game lost and leaves the last valid body intact. The current tail
        cell is free to enter if this move does not eat food.
        """
        if action is not None and not isinstance(action, Direction):
            raise TypeError("action must be a Direction or None")
        if self.status is not GameStatus.RUNNING:
            return self.status
        if action is not None and not self.direction.is_opposite(action):
            self.direction = action
        dx, dy = self.direction.value
        head_x, head_y = self.snake[0]
        next_head = (head_x + dx, head_y + dy)
        eating = next_head == self.food
        body = self.snake if eating else self.snake[:-1]
        x, y = next_head
        if not (0 <= x < self.width and 0 <= y < self.height) or next_head in body:
            self.status = GameStatus.LOST
            return self.status
        self.snake.insert(0, next_head)
        if eating:
            self.score += 1
            self.food = self._spawn_food()
            if self.food is None:
                self.status = GameStatus.WON
        else:
            self.snake.pop()
        return self.status
