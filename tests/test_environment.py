"""Headless observations, actions, rewards, and episode boundaries."""

from array import array
from copy import deepcopy
import subprocess
import sys

import pytest

from snake_ai.game import Direction, GameStatus
from snake_ai.rl import FEATURE_NAMES, EnvConfig, RewardConfig, SnakeEnv, relative_direction


def make_env(**kwargs):
    env = SnakeEnv(**kwargs)
    env.reset(seed=42)
    env.game.food = (0, 0)
    return env


def test_feature_order_and_numeric_array():
    assert FEATURE_NAMES == (
        "danger_straight", "danger_right", "danger_left", "moving_up", "moving_down",
        "moving_left", "moving_right", "food_left", "food_right", "food_above", "food_below")
    env = make_env()
    obs = env.observation()
    assert isinstance(obs, array) and obs.typecode == "f" and obs.itemsize == 4
    assert len(obs) == 11 and set(obs) <= {0.0, 1.0}
    assert list(obs) == [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    obs[0] = 1
    assert env.observation()[0] == 0  # Returned observations are independent copies.


@pytest.mark.parametrize("direction, expected", [
    (Direction.UP, [1, 0, 0, 0]), (Direction.DOWN, [0, 1, 0, 0]),
    (Direction.LEFT, [0, 0, 1, 0]), (Direction.RIGHT, [0, 0, 0, 1]),
])
def test_heading_features(direction, expected):
    env = make_env()
    env.game.direction = direction
    assert list(env.observation()[3:7]) == expected


@pytest.mark.parametrize("food, expected", [
    ((9, 10), [1, 0, 0, 0]), ((11, 10), [0, 1, 0, 0]),
    ((10, 9), [0, 0, 1, 0]), ((10, 11), [0, 0, 0, 1]),
    ((9, 9), [1, 0, 1, 0]), ((11, 11), [0, 1, 0, 1]),
    (None, [0, 0, 0, 0]),
])
def test_food_features(food, expected):
    env = make_env()
    env.game.food = food
    assert list(env.observation()[7:]) == expected


@pytest.mark.parametrize("direction, expected", [
    (Direction.UP, [Direction.UP, Direction.RIGHT, Direction.LEFT]),
    (Direction.RIGHT, [Direction.RIGHT, Direction.DOWN, Direction.UP]),
    (Direction.DOWN, [Direction.DOWN, Direction.LEFT, Direction.RIGHT]),
    (Direction.LEFT, [Direction.LEFT, Direction.UP, Direction.DOWN]),
])
def test_all_relative_actions(direction, expected):
    for action, absolute in enumerate(expected):
        assert relative_direction(direction, action) is absolute
        env = make_env()
        env.game.snake = [(10, 10)]
        env.game.direction = direction
        env.step(action)
        dx, dy = absolute.value
        assert env.game.snake[0] == (10 + dx, 10 + dy)


@pytest.mark.parametrize("action", [-1, 3, 1.0, True, False, "1", None, [1, 0, 0]])
def test_invalid_action_rejected_without_mutation(action):
    env = make_env()
    before = (env.game.snake.copy(), env.game.direction, env.steps)
    with pytest.raises(ValueError):
        env.step(action)
    assert (env.game.snake, env.game.direction, env.steps) == before


@pytest.mark.parametrize("direction, head", [
    (Direction.UP, (0, 0)), (Direction.RIGHT, (19, 0)),
    (Direction.DOWN, (19, 19)), (Direction.LEFT, (0, 19)),
])
def test_wall_danger_matches_actual_moves(direction, head):
    env = make_env()
    env.game.direction = direction
    env.game.snake = [head]
    assert list(env.observation()[:3]) == [1, 0, 1]
    for action in range(3):
        clone = deepcopy(env.game)
        status = clone.step(relative_direction(direction, action))
        assert bool(env.observation()[action]) == (status is GameStatus.LOST)


@pytest.mark.parametrize("turns", range(4))
def test_body_danger_and_departing_tail_for_all_headings(turns):
    env = make_env()
    points = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    def rotate(x, y):
        for _ in range(turns):
            x, y = -y, x
        return (10 + x, 10 + y)
    env.game.snake = [rotate(x, y) for x, y in points]
    direction = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT][turns]
    env.game.direction = direction
    # Straight enters the departing tail; right enters an occupied body segment.
    assert list(env.observation()[:3]) == [0, 1, 0]
    before = (env.game.snake.copy(), env.game._rng.getstate())
    for action in range(3):
        clone = deepcopy(env.game)
        status = clone.step(relative_direction(direction, action))
        assert bool(env.observation()[action]) == (status is GameStatus.LOST)
    assert (env.game.snake, env.game._rng.getstate()) == before


def test_ordinary_move_and_eating_rewards():
    env = make_env()
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["steps_since_food"] == 1
    env.game.food = (12, 10)
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (10.0, False, False)
    assert info["score"] == 1 and info["steps"] == 2
    assert info["steps_since_food"] == 0
    assert len(env.game.snake) == 4


def test_collision_termination_takes_precedence_over_limits():
    env = make_env(config=EnvConfig(max_steps=1, starvation_min_steps=1, starvation_factor=1))
    env.game.snake = [(19, 10)]
    obs, reward, terminated, truncated, info = env.step(0)
    assert len(obs) == 11
    assert (reward, terminated, truncated) == (-10.0, True, False)
    assert info["end_reason"] == "collision" and info["game_status"] == "lost"
    before = env.game.snake.copy()
    with pytest.raises(RuntimeError):
        env.step(0)
    assert env.game.snake == before and env.steps == 1


def test_win_reward_replaces_food_reward_and_limits():
    env = make_env(config=EnvConfig(width=2, height=2, max_steps=1))
    env.game.snake = [(0, 0), (1, 0), (1, 1)]
    env.game.direction = Direction.LEFT
    env.game.food = (0, 1)
    obs, reward, terminated, truncated, info = env.step(2)
    assert (reward, terminated, truncated) == (10.0, True, False)
    assert info["end_reason"] == "win" and info["game_status"] == "won"
    assert info["score"] == 1 and list(obs[7:]) == [0, 0, 0, 0]


def test_max_steps_truncation_keeps_movement_reward():
    env = make_env(config=EnvConfig(max_steps=2))
    assert env.step(0)[2:4] == (False, False)
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["end_reason"] == "max_steps" and info["game_status"] == "running"
    with pytest.raises(RuntimeError):
        env.step(0)


def test_starvation_truncation_at_exact_limit():
    env = make_env(config=EnvConfig(starvation_min_steps=4, starvation_factor=1))
    for _ in range(3):
        assert env.step(1)[2:4] == (False, False)
    _, reward, terminated, truncated, info = env.step(1)
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["end_reason"] == "starvation" and info["steps_since_food"] == 4


def test_food_resets_starvation_and_new_length_changes_allowance():
    env = make_env(config=EnvConfig(starvation_min_steps=2, starvation_factor=1))
    assert env.starvation_limit == 3
    env.step(0)
    env.step(0)
    env.game.food = (13, 10)
    _, reward, terminated, truncated, info = env.step(0)
    assert (reward, terminated, truncated) == (10.0, False, False)
    assert info["steps_since_food"] == 0 and info["starvation_limit"] == 4


def test_eating_at_step_cap_is_truncated_with_food_reward():
    env = make_env(config=EnvConfig(max_steps=1))
    env.game.food = (11, 10)
    assert env.step(0)[1:4] == (10.0, False, True)


def test_configurable_rewards():
    env = make_env(rewards=RewardConfig(food=3, movement=0.25, collision=-7, win=11))
    assert env.step(0)[1] == 0.25
    env.game.food = (12, 10)
    assert env.step(0)[1] == 3.0
    env.game.snake = [(19, 10)]
    assert env.step(0)[1] == -7.0
    # Use a small board for the win configuration check.
    env = SnakeEnv(EnvConfig(width=2, height=2), env.rewards)
    env.reset(seed=3)
    env.game.snake = [(0, 0), (1, 0), (1, 1)]
    env.game.direction = Direction.LEFT
    env.game.food = (0, 1)
    assert env.step(2)[1] == 11.0


def test_reset_reproducibility_and_counters():
    env = SnakeEnv(config=EnvConfig(max_steps=1))
    with pytest.raises(RuntimeError):
        env.step(0)
    first = env.reset(seed=7)
    first_food = env.game.food
    env.step(0)
    assert env.reset(seed=7) == first
    assert env.game.food == first_food
    assert env.steps == env.steps_since_food == 0
    assert env.game.status is GameStatus.RUNNING
    assert env.step(0)[3] is True


def test_matching_episodes_and_reset_without_seed():
    first, second = SnakeEnv(), SnakeEnv()
    assert first.reset(seed=12) == second.reset(seed=12)
    for _ in range(20):
        assert first.step(1) == second.step(1)
    assert first.reset() == second.reset()


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"starvation_factor": 0},
                                     {"starvation_min_steps": True}, {"width": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "10", True])
def test_invalid_reward(value):
    with pytest.raises(ValueError):
        RewardConfig(food=value)


def test_headless_evaluation_never_imports_rendering_or_ml():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from snake_ai.evaluate import evaluate; evaluate(episodes=2); "
        "assert not ({'pygame', 'torch', 'numpy', 'gymnasium'} & sys.modules.keys())"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
