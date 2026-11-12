"""Automated Pygame smoke checks using an invisible SDL display."""

from types import SimpleNamespace

import pytest

from snake_ai.game import Direction, GameStatus, SnakeGame


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from snake_ai.game import renderer
    yield renderer
    renderer.pygame.quit()


@pytest.mark.parametrize("status", list(GameStatus))
def test_draw_board_and_terminal_screens(renderer, status):
    pg = renderer.pygame
    pg.init()
    screen = pg.display.set_mode((520, 584))
    game = SnakeGame(seed=7)
    game.status = status
    if status is GameStatus.WON:
        game.food = None
    renderer.draw(screen, game, pg.font.Font(None, 30), pg.font.Font(None, 22))
    pg.display.flip()
    assert screen.get_size() == (520, 584)


@pytest.mark.parametrize("exit_key", ["escape", "close"])
def test_event_loop_turn_restart_and_exit(renderer, monkeypatch, exit_key):
    pg = renderer.pygame
    game = SnakeGame(seed=7)
    game.food = (0, 0)
    key = lambda value: pg.event.Event(pg.KEYDOWN, key=value)
    exit_event = key(pg.K_ESCAPE) if exit_key == "escape" else pg.event.Event(pg.QUIT)
    batches = iter([
        [key(pg.K_UP), key(pg.K_LEFT)],  # Only one turn is accepted per move.
        [key(pg.K_r)],
        [exit_event],
    ])
    observed = []
    original_draw = renderer.draw

    def record_draw(screen, state, font, small_font):
        observed.append((state.snake[0], state.direction))
        original_draw(screen, state, font, small_font)

    monkeypatch.setattr(pg.event, "get", lambda: next(batches))
    monkeypatch.setattr(pg.time, "Clock", lambda: SimpleNamespace(tick=lambda fps: 100))
    monkeypatch.setattr(renderer, "draw", record_draw)
    renderer.run(game)
    assert observed == [((10, 9), Direction.UP), ((10, 10), Direction.RIGHT)]
    assert not pg.get_init()
