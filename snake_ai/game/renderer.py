"""Pygame display, keyboard input, and timing for the independent engine."""

import math
import pygame

from .engine import Direction, GameStatus, SnakeGame

CELL_SIZE = 26
HEADER_HEIGHT = 64
BACKGROUND = (17, 23, 30)
TEXT = (234, 240, 246)


def draw(screen: pygame.Surface, game: SnakeGame,
         font: pygame.font.Font, small_font: pygame.font.Font, *,
         controls_text: str = "Arrows: move   R: restart   Esc: exit",
         show_terminal_overlay: bool = True) -> None:
    screen.fill(BACKGROUND)
    score_text = font.render(f"SnakeAI   |   Score: {game.score}", True, TEXT)
    screen.blit(score_text, (12, 8))
    screen.blit(small_font.render(controls_text, True,
                                (162, 177, 192)), (12, 38))
    for x in range(game.width + 1):
        pygame.draw.line(screen, (28, 36, 46), (x * CELL_SIZE, HEADER_HEIGHT),
                         (x * CELL_SIZE, screen.get_height()))
    for y in range(game.height + 1):
        row_y = HEADER_HEIGHT + y * CELL_SIZE
        pygame.draw.line(screen, (28, 36, 46), (0, row_y), (screen.get_width(), row_y))

    def cell(position: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = position
        rect = pygame.Rect(x * CELL_SIZE + 2, HEADER_HEIGHT + y * CELL_SIZE + 2,
                           CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(screen, color, rect, border_radius=5)

    if game.food is not None:
        cell(game.food, (255, 113, 107))
    for index, position in enumerate(game.snake):
        cell(position, (139, 244, 179) if index == 0 else (56, 185, 126))
    if show_terminal_overlay and game.status is not GameStatus.RUNNING:
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((8, 12, 18, 215))
        screen.blit(overlay, (0, 0))
        title = "You filled the board!" if game.status is GameStatus.WON else "Game over"
        lines = [font.render(title, True, TEXT),
                 font.render(f"Final score: {game.score}", True, TEXT),
                 small_font.render("Press R to restart or Esc to exit", True, TEXT)]
        center_y = screen.get_height() // 2
        for index, line in enumerate(lines):
            screen.blit(line, line.get_rect(center=(screen.get_width() // 2,
                                                    center_y + (index - 1) * 36)))


def run(game: SnakeGame, moves_per_second: float = 10) -> None:
    if not math.isfinite(moves_per_second) or not 0 < moves_per_second <= 60:
        raise ValueError("Movement speed must be greater than 0 and at most 60")
    pygame.init()
    try:
        screen = pygame.display.set_mode((game.width * CELL_SIZE,
                                          game.height * CELL_SIZE + HEADER_HEIGHT))
        pygame.display.set_caption("SnakeAI - Milestone 1")
        font = pygame.font.Font(None, 30)
        small_font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        keys = {pygame.K_UP: Direction.UP, pygame.K_DOWN: Direction.DOWN,
                pygame.K_LEFT: Direction.LEFT, pygame.K_RIGHT: Direction.RIGHT}
        pending: Direction | None = None
        elapsed = 0.0
        interval = 1.0 / moves_per_second
        running = True
        while running:
            # Cap catch-up after window dragging or debugger pauses.
            elapsed += min(clock.tick(60) / 1000, 0.25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                        pending = None
                        elapsed = 0.0
                    elif (event.key in keys and pending is None
                          and game.status is GameStatus.RUNNING):
                        action = keys[event.key]
                        if not game.direction.is_opposite(action):
                            pending = action
            if not running:
                break
            while elapsed >= interval and game.status is GameStatus.RUNNING:
                game.step(pending)
                pending = None
                elapsed -= interval
            if game.status is not GameStatus.RUNNING:
                elapsed = 0.0
                pending = None
            draw(screen, game, font, small_font)
            pygame.display.flip()
    finally:
        pygame.quit()
