"""Watch a saved DQN play greedily. This demo never trains or saves weights."""

import argparse
import math
from pathlib import Path

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import FEATURE_NAMES, SnakeEnv

DEFAULT_CHECKPOINT = Path("results/milestone4_initial/checkpoints/final.pt")
ACTION_NAMES = ("Straight", "Right", "Left")
MIN_SPEED = 1.0
MAX_SPEED = 60.0
MARGIN = 20
PANEL_WIDTH = 360
MIN_HEIGHT = 700


class PlaybackSession:
    """Inference and episode controls, independent of the Pygame event loop.

    During play the cached Q-values describe the current observation and the
    next action. At an episode end they retain the values before the final
    move, explicitly labelled as such in the panel.
    """

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, *, seed: int = 800000,
                 speed: float = 10.0) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED):
            raise ValueError("speed must be between 1 and 60 moves per second")
        self.checkpoint = Path(checkpoint)
        self.agent = DQNAgent.load(self.checkpoint)
        self.model = self.agent.online.cpu().eval()
        self.model.requires_grad_(False)
        if (self.model.input_layer.in_features != len(FEATURE_NAMES)
                or self.model.output_layer.out_features != len(ACTION_NAMES)):
            raise ValueError("Checkpoint must accept 11 observations and produce 3 relative action values")
        # Reuse every environment setting from the checkpoint; no demo overrides.
        self.env = SnakeEnv(self.agent.config.environment, self.agent.config.rewards)
        self.base_seed = seed
        self.episode = 1
        self.speed = float(speed)
        self.paused = False
        self.reset()

    @property
    def seed(self) -> int:
        return self.base_seed + self.episode - 1

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def _refresh_decision(self) -> None:
        if len(self.observation) != self.model.input_layer.in_features:
            raise ValueError("Environment observation size does not match checkpoint")
        inputs = torch.tensor(list(self.observation), dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            outputs = self.model(inputs)
            if outputs.shape != (3,) or not torch.isfinite(outputs).all():
                raise ValueError("Checkpoint must produce three finite Q-values")
            self.q_values = tuple(float(value) for value in outputs.tolist())
            self.selected_action = int(outputs.argmax(dim=-1).item())

    def reset(self, *, next_episode: bool = False) -> None:
        """R retries the same seed; N increments episode/seed and starts fresh."""
        if next_episode:
            self.episode += 1
        self.observation, self.info = self.env.reset(seed=self.seed)
        self.terminated = False
        self.truncated = False
        self.paused = False
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_action: int | None = None
        self._refresh_decision()

    def step(self) -> bool:
        """Execute the cached argmax through the original environment API."""
        if self.paused or self.done:
            return False
        self.last_action = self.selected_action
        (self.observation, self.last_reward, self.terminated,
         self.truncated, self.info) = self.env.step(self.last_action)
        self.total_reward += self.last_reward
        if not self.done:
            self._refresh_decision()
        return True

    def change_speed(self, delta: float) -> None:
        self.speed = max(MIN_SPEED, min(MAX_SPEED, self.speed + delta))


def window_size(session: PlaybackSession) -> tuple[int, int]:
    from snake_ai.game.renderer import CELL_SIZE, HEADER_HEIGHT
    board_width = session.env.game.width * CELL_SIZE
    board_height = session.env.game.height * CELL_SIZE + HEADER_HEIGHT
    return board_width + PANEL_WIDTH + 3 * MARGIN, max(board_height + 2 * MARGIN + 64, MIN_HEIGHT)


def draw(screen, session: PlaybackSession, fonts: dict) -> None:
    """Reuse the game renderer on a separate surface, plus an inference panel."""
    import pygame
    from snake_ai.game import renderer

    background = (12, 18, 25)
    panel_color = (22, 31, 41)
    text_color = (235, 242, 247)
    muted = (163, 181, 196)
    accent = (139, 244, 179)
    screen.fill(background)
    board_width = session.env.game.width * renderer.CELL_SIZE
    board_height = session.env.game.height * renderer.CELL_SIZE + renderer.HEADER_HEIGHT
    board = pygame.Surface((board_width, board_height))
    renderer.draw(board, session.env.game, fonts["body"], fonts["small"],
                  controls_text="Trained DQN   |   Greedy CPU inference",
                  show_terminal_overlay=False)
    screen.blit(board, (MARGIN, MARGIN))
    panel = pygame.Rect(board_width + 2 * MARGIN, MARGIN, PANEL_WIDTH, screen.get_height() - 2 * MARGIN)
    pygame.draw.rect(screen, panel_color, panel, border_radius=12)
    left = panel.x + 20
    available_width = PANEL_WIDTH - 40

    def text(value: str, x: int, y: int, *, font="body", color=text_color, width=None):
        face = fonts[font]
        if width is not None and face.size(value)[0] > width:
            while value and face.size(value + "...")[0] > width:
                value = value[:-1]
            value += "..."
        screen.blit(face.render(value, True, color), (x, panel.y + y))

    text("TRAINED AGENT", left, 18, font="small", color=accent)
    text("SnakeAI", left, 40, font="title")
    state_label = "EPISODE ENDED" if session.done else "PAUSED" if session.paused else "PLAYING"
    text(f"{state_label}  |  {session.speed:g} moves/sec", left, 80, font="small", color=muted)
    pygame.draw.line(screen, (49, 64, 77), (left, panel.y + 112), (panel.right - 20, panel.y + 112))
    text("CHECKPOINT", left, 130, font="small", color=muted)
    text(session.checkpoint.name, left, 150, width=available_width)
    text(f"Episode {session.episode}   |   Seed {session.seed}", left, 184,
         font="small", color=muted, width=available_width)
    text("SCORE", left, 218, font="small", color=muted)
    text("MOVES", left + 166, 218, font="small", color=muted)
    text(str(session.info["score"]), left, 238, font="title")
    text(str(session.info["steps"]), left + 166, 238, font="title")
    text(f"Direction: {session.env.game.direction.name}", left, 286)
    text(f"Last reward: {session.last_reward:+g}", left, 316, font="small", color=muted)
    text(f"Episode reward: {session.total_reward:+g}", left, 338, font="small", color=muted)
    text("FINAL MOVE Q-VALUES" if session.done else "NEXT ACTION Q-VALUES",
         left, 376, font="small", color=muted)
    for action, name in enumerate(ACTION_NAMES):
        y = 402 + action * 45
        selected = action == session.selected_action
        rect = pygame.Rect(left, panel.y + y, available_width, 38)
        pygame.draw.rect(screen, (36, 71, 57) if selected else (30, 42, 54), rect, border_radius=6)
        if selected:
            pygame.draw.rect(screen, accent, rect, width=1, border_radius=6)
        text(("> " if selected else "   ") + name, left + 10, y + 8,
             color=accent if selected else text_color)
        value = fonts["body"].render(f"{session.q_values[action]:+.3f}", True,
                                    accent if selected else text_color)
        screen.blit(value, (rect.right - value.get_width() - 12, rect.y + 8))
    text(f"{'Last action' if session.done else 'Selected'}: {ACTION_NAMES[session.selected_action]} ({session.selected_action})",
         left, 548, color=accent)
    text("Values before final move" if session.done else "Values for the displayed board",
         left, 578, font="small", color=muted)
    text("Q estimates future discounted reward.", left, 607, font="small", color=muted)
    text("Values are not probabilities.", left, 629, font="small", color=muted)

    # Controls remain below the game, so arrow keys never suggest human steering.
    footer_y = MARGIN + board_height + 16
    for line in ("Space: pause/resume    R: retry    N: next episode",
                 "+ / -: speed    Esc: exit"):
        screen.blit(fonts["small"].render(line, True, muted), (MARGIN, footer_y))
        footer_y += 24
    if session.paused or session.done:
        overlay = pygame.Surface((board_width, board_height - renderer.HEADER_HEIGHT), pygame.SRCALPHA)
        overlay.fill((7, 12, 18, 205))
        screen.blit(overlay, (MARGIN, MARGIN + renderer.HEADER_HEIGHT))
        if session.done:
            heading = "You filled the board!" if session.info["end_reason"] == "win" else "Episode ended"
            lines = [heading, f"Reason: {session.info['end_reason']}", "R: retry this seed   N: next episode"]
        else:
            lines = ["Paused", "Press Space to resume"]
        for index, line in enumerate(lines):
            face = fonts["body"] if index == 0 else fonts["small"]
            rendered = face.render(line, True, text_color)
            center = (MARGIN + board_width // 2, MARGIN + board_height // 2 + (index - 1) * 34)
            screen.blit(rendered, rendered.get_rect(center=center))


def run(session: PlaybackSession) -> None:
    import pygame

    pygame.init()
    try:
        screen = pygame.display.set_mode(window_size(session))
        pygame.display.set_caption("SnakeAI - Trained AI playback")
        fonts = {"title": pygame.font.Font(None, 38), "body": pygame.font.Font(None, 25),
                 "small": pygame.font.Font(None, 21)}
        clock = pygame.time.Clock()
        elapsed = 0.0
        running = True
        while running:
            delta = min(clock.tick(60) / 1000, 0.25)
            # Dropping elapsed time while paused/ended avoids a catch-up burst.
            elapsed = 0.0 if session.paused or session.done else elapsed + delta
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not session.done:
                            session.paused = not session.paused
                        elapsed = 0.0
                    elif event.key in (pygame.K_r, pygame.K_n):
                        session.reset(next_episode=event.key == pygame.K_n)
                        elapsed = 0.0
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        session.change_speed(1)
                        elapsed = 0.0
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        session.change_speed(-1)
                        elapsed = 0.0
                    # Arrow keys are ignored; only the network chooses directions.
            if not running:
                break
            interval = 1.0 / session.speed
            while elapsed >= interval and not session.paused and not session.done:
                session.step()
                elapsed -= interval
            draw(screen, session, fonts)
            pygame.display.flip()
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=800000)
    parser.add_argument("--speed", type=float, default=10.0, help="Moves per second, from 1 to 60")
    args = parser.parse_args()
    try:
        session = PlaybackSession(args.checkpoint, seed=args.seed, speed=args.speed)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(f"Could not prepare trained AI playback: {error}")
    run(session)


if __name__ == "__main__":
    main()
