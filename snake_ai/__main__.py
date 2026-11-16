"""Run with python -m snake_ai, or the snake-ai console command."""

import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description="Play SnakeAI with keyboard controls")
    parser.add_argument("--speed", type=float, default=10,
                        help="Moves per second (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional reproducible food seed")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 60:
        parser.error("--speed must be greater than 0 and at most 60")
    from .game.engine import SnakeGame
    from .game.renderer import run
    run(SnakeGame(seed=args.seed), moves_per_second=args.speed)


if __name__ == "__main__":
    main()
