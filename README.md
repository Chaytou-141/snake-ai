# SnakeAI

An educational Python project that will eventually explore Deep Q-Networks with
PyTorch. **Milestone 1** builds only the playable, tested Snake game. There is no
agent, training algorithm, or neural network in this milestone.

## Setup on Windows

Requires Python 3.11 or newer. Open this folder in VS Code (rather than the
neighboring electricity project). The initial environment was created with Python
3.12. Run these PowerShell commands to reproduce the setup:

```powershell
Set-Location 'C:\Users\ilach\Desktop\VSCODE\snake-ai'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

The editable install includes Pygame; the `dev` extra includes pytest. No PyTorch
is needed. Using the interpreter directly avoids PowerShell activation-policy
issues. VS Code settings select `.venv` and enable pytest discovery. Install the
Microsoft Python and Python Debugger extensions for editor testing and F5 launch.
If VS Code retains another interpreter, use **Python: Select Interpreter** and
choose `.venv\Scripts\python.exe`.

## Play

```powershell
.\.venv\Scripts\python.exe -m snake_ai
.\.venv\Scripts\python.exe -m snake_ai --seed 42 --speed 10
```

Arrow keys steer, **R** restarts at any time, and **Escape** or the window close
button exits. Movement defaults to 10 cells per second; `--speed` accepts values
greater than zero and up to 60. The display refreshes independently at up to 60 FPS.
The first valid arrow received between movement steps is used for the next move.

## Rules and engine

The default board has 20 columns and 20 rows. Coordinates `(x, y)` start at the
upper-left corner; x grows rightward and y grows downward. The snake begins at
`[(10, 10), (9, 10), (8, 10)]`, with its head first, facing right.

`SnakeGame.step(action)` moves one cell. An action is a `Direction`, or `None` to
continue forward. An immediate reverse turn is ignored. Eating food keeps the
tail in place, grows the body by one, and adds one point. Walls and the body cause
a loss. **Entering the current tail cell is legal when the tail moves away on
that step.** A collision leaves the body at its last valid position.

Food is chosen uniformly from an explicit list of empty cells. Filling the grid
sets `GameStatus.WON` and `food = None`, so food generation never loops forever.
Calling `step()` after a win or loss leaves the state unchanged until reset.
Custom board dimensions of at least 2 by 2 are supported by the engine; the
interactive layout is designed for the default 20 by 20 board.

Each game owns a `random.Random` instance. Matching seeds and action/reset
sequences produce matching states. `reset()` resets the board while continuing
the RNG stream; `reset(seed=42)` also restarts that seed's food sequence. The seed
does not control the player's keyboard timing.

```python
from snake_ai.game import Direction, SnakeGame

game = SnakeGame(seed=42)
status = game.step(Direction.UP)
print(game.snake, game.food, game.score, status)
game.reset(seed=42)
```

The engine uses only the standard library. `renderer.run()` collects keyboard
input, translates it into directions, and calls `step()` on a movement timer.
`renderer.draw()` reads state to display the board, score, and terminal overlay.
Keeping input, timing, and drawing outside the rules lets tests and future
training run without opening windows.

## Tests

```powershell
# Focused checks, followed by the complete suite
.\.venv\Scripts\python.exe -m pytest tests/test_engine.py -k 'collision or tail or full_grid' -q
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use fixed seeds and explicit board fixtures. They cover starting state,
movement, turns and reversals, eating, growth, scoring, wall and self collisions,
food placement, resets, reproducibility, tail movement, full boards, validation,
and importing the engine without Pygame. No graphical window is required.

`tests/test_renderer.py` also uses SDL's dummy display to smoke-test drawing the
running, lost, and won screens, plus queued turns, restart, and both exit paths.
These automated checks do not replace a manual display and keyboard play-through.

Validation on 12 November 2026 with Python 3.12, Pygame 2.6.1, and pytest 9.2.0:
focused rule checks **7 passed, 21 deselected**; complete suite **33 passed**.
The interactive game has not been manually played or visually inspected.

## Project structure

```text
snake-ai/
  snake_ai/
    __init__.py
    __main__.py
    game/
      __init__.py
      engine.py
      renderer.py
  tests/
    test_engine.py
    test_renderer.py
  .vscode/
    settings.json
    launch.json
  .gitignore
  pyproject.toml
  README.md
  LEARNING_JOURNAL.md
```

`.venv/` and generated installation/test caches are local, ignored artifacts.
Dependencies have version ranges rather than a fully pinned lock file. This
project has no Git repository initialized automatically; no commit or push is
part of the setup.

## Later milestones

A future adapter can translate agent outputs to `Direction`, read observations
from engine state, calculate rewards, and use terminal status to end episodes.
The same `step()` and `reset()` rules will serve both human play and training.
Observation encoding, rewards, replay memory, DQN, and PyTorch are deliberately
left for later milestones. State attributes are exposed for learning and reading;
callers should use `step()` and `reset()` to keep normal gameplay consistent.
