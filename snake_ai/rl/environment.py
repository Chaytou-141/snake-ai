"""Gymnasium-style API using only the Python standard library."""

from array import array
from dataclasses import dataclass
import math

from snake_ai.game import Direction, GameStatus, SnakeGame
from snake_ai.game.geometry import free_cell_distance, body_free_cell_distance

FEATURE_NAMES = (
    "danger_straight", "danger_right", "danger_left",
    "moving_up", "moving_down", "moving_left", "moving_right",
    "food_left", "food_right", "food_above", "food_below",
)
_CLOCKWISE = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)
DISTANCE_FEATURE_NAMES = ("distance_straight", "distance_right", "distance_left")
FOOD_DISTANCE_FEATURE_NAMES = ("food_dx", "food_dy")
BODY_DISTANCE_FEATURE_NAMES = ("body_distance_straight", "body_distance_right", "body_distance_left")


def feature_names(observation_mode: str = "original11") -> tuple[str, ...]:
    if observation_mode == "original11":
        return FEATURE_NAMES
    if observation_mode == "distance14":
        return FEATURE_NAMES + DISTANCE_FEATURE_NAMES
    if observation_mode == "food13":
        return FEATURE_NAMES + FOOD_DISTANCE_FEATURE_NAMES
    if observation_mode == "body16":
        return FEATURE_NAMES + FOOD_DISTANCE_FEATURE_NAMES + BODY_DISTANCE_FEATURE_NAMES
    raise ValueError("observation_mode must be 'original11', 'distance14', 'food13', or 'body16'")


def relative_direction(direction: Direction, action: int) -> Direction:
    """0 = straight, 1 = clockwise/right, 2 = counterclockwise/left."""
    if type(action) is not int or action not in (0, 1, 2):
        raise ValueError("action must be the integer 0, 1, or 2 (not bool)")
    return _CLOCKWISE[(_CLOCKWISE.index(direction) + (0, 1, -1)[action]) % 4]


@dataclass(frozen=True)
class RewardConfig:
    food: float = 10.0
    collision: float = -10.0
    movement: float = 0.0
    win: float = 10.0

    def __post_init__(self) -> None:
        for value in (self.food, self.collision, self.movement, self.win):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("rewards must be finite numbers")


@dataclass(frozen=True)
class EnvConfig:
    width: int = 20
    height: int = 20
    max_steps: int = 10_000
    starvation_min_steps: int = 400
    starvation_factor: int = 100

    def __post_init__(self) -> None:
        for name in ("width", "height", "max_steps", "starvation_min_steps", "starvation_factor"):
            value = getattr(self, name)
            minimum = 2 if name in ("width", "height") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


class SnakeEnv:
    """Wrap the engine. Call reset() before stepping and after an episode ends.

    Observations are fresh array('f') objects. The default 11 binary features
    are unchanged; distance14 appends three normalised static distances.
    food13 appends signed horizontal/vertical offsets to current visible food.
    body16 adds static body-only distances with normalised 1 for no body hit.
    No NumPy, Gymnasium, Pygame, or PyTorch import is required.
    """

    def __init__(self, config: EnvConfig | None = None,
                 rewards: RewardConfig | None = None, *,
                 observation_mode: str = "original11") -> None:
        self.feature_names = feature_names(observation_mode)
        self.observation_mode = observation_mode
        self.config = config if config is not None else EnvConfig()
        self.rewards = rewards if rewards is not None else RewardConfig()
        self.game = SnakeGame(self.config.width, self.config.height)
        self.steps = 0
        self.steps_since_food = 0
        self._needs_reset = True
        self._reason: str | None = None

    @property
    def starvation_limit(self) -> int:
        return max(self.config.starvation_min_steps,
                   self.config.starvation_factor * len(self.game.snake))

    def observation(self) -> array:
        game = self.game
        head_x, head_y = game.snake[0]
        food = game.food
        features = [game.would_collide(relative_direction(game.direction, action))
                    for action in range(3)]
        features.extend(game.direction is direction for direction in
                        (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT))
        features.extend((food is not None and food[0] < head_x,
                         food is not None and food[0] > head_x,
                         food is not None and food[1] < head_y,
                         food is not None and food[1] > head_y))
        if self.observation_mode == "distance14":
            divisor = max(game.width, game.height) - 1
            features.extend(free_cell_distance(game, relative_direction(game.direction, action)) / divisor
                            for action in range(3))
        elif self.observation_mode in ("food13", "body16"):
            features.extend(((food[0] - head_x) / (game.width - 1),
                             (food[1] - head_y) / (game.height - 1))
                            if food is not None else (0.0, 0.0))
            if self.observation_mode == "body16":
                divisor = max(game.width, game.height) - 1
                features.extend(body_free_cell_distance(game, relative_direction(game.direction, action)) / divisor
                                for action in range(3))
        return array("f", features)

    def _info(self) -> dict[str, object]:
        return {"score": self.game.score, "steps": self.steps,
                "steps_since_food": self.steps_since_food,
                "starvation_limit": self.starvation_limit,
                "game_status": self.game.status.value, "end_reason": self._reason}

    def reset(self, seed: int | None = None) -> tuple[array, dict[str, object]]:
        """Explicit seeds restart food RNG; None continues its existing stream."""
        self.game.reset(seed=seed)
        self.steps = 0
        self.steps_since_food = 0
        self._reason = None
        self._needs_reset = False
        return self.observation(), self._info()

    def step(self, action: int) -> tuple[array, float, bool, bool, dict[str, object]]:
        direction = relative_direction(self.game.direction, action)
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping a new episode")
        old_score = self.game.score
        status = self.game.step(direction)
        self.steps += 1
        ate = self.game.score > old_score
        self.steps_since_food = 0 if ate else self.steps_since_food + 1
        terminated = status is not GameStatus.RUNNING
        truncated = False
        if status is GameStatus.LOST:
            reward, self._reason = self.rewards.collision, "collision"
        elif status is GameStatus.WON:
            # Win replaces the food reward: the final meal is not double counted.
            reward, self._reason = self.rewards.win, "win"
        else:
            reward = self.rewards.food if ate else self.rewards.movement
            if self.steps >= self.config.max_steps:
                truncated, self._reason = True, "max_steps"
            elif self.steps_since_food >= self.starvation_limit:
                truncated, self._reason = True, "starvation"
        self._needs_reset = terminated or truncated
        return self.observation(), float(reward), terminated, truncated, self._info()
