"""Headless rule tests: custom boards are deliberate, deterministic fixtures."""

import random
import subprocess
import sys

import pytest

from snake_ai.game import Direction, GameStatus, SnakeGame


def test_starting_position():
    game = SnakeGame(seed=7)
    assert (game.width, game.height) == (20, 20)
    assert game.snake == [(10, 10), (9, 10), (8, 10)]
    assert game.direction is Direction.RIGHT
    assert game.score == 0
    assert game.status is GameStatus.RUNNING
    assert game.food not in game.snake


def test_forward_movement():
    game = SnakeGame(seed=7)
    game.food = (0, 0)
    assert game.step() is GameStatus.RUNNING
    assert game.snake == [(11, 10), (10, 10), (9, 10)]
    assert game.score == 0


@pytest.mark.parametrize("action, head", [(Direction.UP, (10, 9)),
                                          (Direction.DOWN, (10, 11))])
def test_valid_turns(action, head):
    game = SnakeGame(seed=7)
    game.food = (0, 0)
    game.step(action)
    assert game.direction is action
    assert game.snake[0] == head


@pytest.mark.parametrize("direction", list(Direction))
def test_reverse_turn_ignored(direction):
    game = SnakeGame(seed=7)
    dx, dy = direction.value
    game.snake = [(10, 10), (10 - dx, 10 - dy), (10 - 2 * dx, 10 - 2 * dy)]
    game.direction = direction
    game.food = (0, 0)
    reverse = next(item for item in Direction if direction.is_opposite(item))
    game.step(reverse)
    assert game.direction is direction
    assert game.snake[0] == (10 + dx, 10 + dy)
    assert game.status is GameStatus.RUNNING


def test_eating_growth_score_and_new_food():
    game = SnakeGame(seed=7)
    game.food = (11, 10)
    old_tail = game.snake[-1]
    game.step()
    assert game.snake[0] == (11, 10)
    assert len(game.snake) == 4
    assert game.snake[-1] == old_tail
    assert game.score == 1
    assert game.food is not None and game.food not in game.snake


@pytest.mark.parametrize("direction, head", [
    (Direction.LEFT, (0, 5)), (Direction.RIGHT, (19, 5)),
    (Direction.UP, (5, 0)), (Direction.DOWN, (5, 19)),
])
def test_wall_collision(direction, head):
    game = SnakeGame(seed=7)
    game.snake = [head]
    game.direction = direction
    assert game.step() is GameStatus.LOST
    assert game.snake == [head]
    assert game.score == 0


def test_self_collision():
    game = SnakeGame(seed=7)
    game.snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)]
    game.direction = Direction.UP
    game.food = (0, 0)
    before = game.snake.copy()
    assert game.step(Direction.RIGHT) is GameStatus.LOST
    assert game.snake == before


def test_moving_into_vacated_tail():
    game = SnakeGame(seed=7)
    game.snake = [(2, 2), (2, 3), (3, 3), (3, 2)]
    game.direction = Direction.UP
    game.food = (0, 0)
    assert game.step(Direction.RIGHT) is GameStatus.RUNNING
    assert game.snake == [(3, 2), (2, 2), (2, 3), (3, 3)]


def test_food_uses_only_remaining_cell():
    game = SnakeGame(width=2, height=2, seed=7)
    game.snake = [(0, 0), (1, 0), (1, 1)]
    assert game._spawn_food() == (0, 1)


def test_full_grid_wins_without_spawning_loop():
    game = SnakeGame(width=2, height=2, seed=7)
    game.snake = [(0, 0), (1, 0), (1, 1)]
    game.direction = Direction.LEFT
    game.food = (0, 1)
    assert game.step(Direction.DOWN) is GameStatus.WON
    assert len(game.snake) == 4
    assert game.score == 1
    assert game.food is None
    assert game._spawn_food() is None
    before = snapshot(game)
    assert game.step() is GameStatus.WON
    assert snapshot(game) == before


def snapshot(game):
    return (game.snake.copy(), game.direction, game.food, game.score, game.status)


def test_lost_game_cannot_move_until_restart():
    game = SnakeGame(seed=7)
    for _ in range(20):
        game.step()
    assert game.status is GameStatus.LOST
    before = snapshot(game)
    assert game.step(Direction.UP) is GameStatus.LOST
    assert snapshot(game) == before


def test_restart_restores_board_and_reseeding_restores_food():
    game = SnakeGame(seed=7)
    initial = snapshot(game)
    game.food = (11, 10)
    game.step()
    for _ in range(20):
        game.step()
    game.reset(seed=7)
    assert snapshot(game) == initial


def test_same_seed_actions_and_restarts_are_deterministic():
    first, second = SnakeGame(seed=42), SnakeGame(seed=42)
    actions = ([Direction.UP] * 3 + [Direction.LEFT] * 3
               + [Direction.DOWN] * 3 + [Direction.RIGHT] * 3)
    for _ in range(3):
        assert snapshot(first) == snapshot(second)
        for action in actions:
            first.step(action)
            second.step(action)
            assert snapshot(first) == snapshot(second)
        first.reset()
        second.reset()


def test_seeded_food_sequence_and_global_rng_independence():
    first, second = SnakeGame(seed=123), SnakeGame(seed=123)
    global_state = random.getstate()
    for _ in range(50):
        assert first._spawn_food() == second._spawn_food()
        assert first._spawn_food() not in first.snake
        second._spawn_food()
    assert random.getstate() == global_state


@pytest.mark.parametrize("width, height", [(1, 20), (20, 1), (0, 0),
                                            (2.5, 20), (True, 20)])
def test_invalid_board_dimensions(width, height):
    with pytest.raises(ValueError):
        SnakeGame(width=width, height=height)


def test_invalid_action_does_not_mutate_state():
    game = SnakeGame(seed=7)
    before = snapshot(game)
    with pytest.raises(TypeError):
        game.step("UP")
    assert snapshot(game) == before


def test_engine_import_does_not_load_pygame():
    result = subprocess.run(
        [sys.executable, "-c",
         "from snake_ai.game import SnakeGame; import sys; "
         "SnakeGame(seed=7).step(); assert 'pygame' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
