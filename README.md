# SnakeAI

An educational Python project that will eventually explore Deep Q-Networks with
PyTorch. **Milestone 1** built the playable, tested Snake game. **Milestone 2**
adds a headless RL environment and a reproducible random-agent baseline. The
random agent does not learn.
**Milestone 3** now adds an untrained PyTorch network and two educational demos.
There is still no DQN algorithm or reinforcement-learning training.

## Setup on Windows

Requires Python 3.11 or newer. Open this folder in VS Code (rather than the
neighboring electricity project). The initial environment was created with Python
3.12. Run these PowerShell commands to reproduce the setup:

```powershell
Set-Location 'C:\Users\ilach\Desktop\VSCODE\snake-ai'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

The editable install includes Pygame; the `dev` extra includes pytest. PyTorch
is needed only for Milestone 3; see its setup below. Using the interpreter
directly avoids PowerShell activation-policy issues. VS Code settings select
`.venv` and enable pytest discovery. Install the
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
Codex did not manually play or inspect the game; the user subsequently reported
manually playing Milestone 1.

## Project structure

```text
snake-ai/
  snake_ai/
    __init__.py
    __main__.py
    evaluate.py
    models/
      __init__.py
      dqn.py
    demos/
      __init__.py
      network.py
      optimisation.py
    game/
      __init__.py
      engine.py
      renderer.py
    rl/
      __init__.py
      environment.py
      random_agent.py
  tests/
    test_engine.py
    test_renderer.py
    test_environment.py
    test_evaluation.py
    test_dqn.py
  results/
    random_baseline.csv
    random_baseline.json
  .vscode/
    settings.json
    launch.json
  .gitignore
  pyproject.toml
  README.md
  LEARNING_JOURNAL.md
```

`.venv/` and generated installation/test caches are local, ignored artifacts.
PyTorch is version-pinned; other dependencies have ranges rather than a fully
pinned lock file.
Milestone 1 was verified with 33 passing tests and checkpointed in Git as
`cee2d5a` before Milestone 2 edits. Milestone 2 was reverified with 88 passing
tests and committed as `77d3151` (`feat: complete SnakeAI reinforcement learning environment`)
before Milestone 3 work. Existing baseline and repeat results were preserved and
verified identical. Milestone 3 remains uncommitted for review. No push was made.

## Milestone 2: headless environment

`SnakeEnv` wraps the existing engine using a Gymnasium-style interface, without
adding Gymnasium, NumPy, PyTorch, or importing Pygame. The renderer and human-play
entry point are unchanged. The engine now exposes `would_collide(direction)`,
also used by `step()`, so danger features use exactly the gameplay collision rules.

```python
from snake_ai.rl import SnakeEnv, RandomAgent

env = SnakeEnv()
agent = RandomAgent(seed=10000)
observation, info = env.reset(seed=42)
while True:
    observation, reward, terminated, truncated, info = env.step(agent.act())
    if terminated or truncated:
        print(info)
        break
```

Call `reset()` before the first action and after every completed episode.
`reset(seed=42)` restarts the food RNG sequence; `reset()` continues its stream.
Reset also clears episode counters and end reason. The agent has a separate RNG:
`agent.reset(seed=10000)` restarts its action sequence; `agent.reset()` preserves
it. Supplying both seeds and the same configurations reproduces an episode.

`step(action)` returns `(observation, reward, terminated, truncated, info)`.
Metadata includes `score`, `steps` (including a fatal move), `steps_since_food`,
`starvation_limit`, `game_status`, and `end_reason`. An invalid action raises
`ValueError` without mutation. Stepping before reset or after episode completion
raises `RuntimeError`. The underlying engine still has its original terminal
no-op behavior. Avoid manipulating `env.game` during real episodes; tests do so
only to arrange controlled scenarios.

### Observation features

Each observation is a fresh standard-library `array('f')`: 11 float32 values,
each 0.0 or 1.0. This avoids a numerical-library dependency and can later be
converted to a PyTorch tensor. The exact zero-based order is:

| Index | Feature |
| --- | --- |
| 0 | Danger straight |
| 1 | Danger right |
| 2 | Danger left |
| 3 | Moving up |
| 4 | Moving down |
| 5 | Moving left |
| 6 | Moving right |
| 7 | Food left |
| 8 | Food right |
| 9 | Food above |
| 10 | Food below |

For `env.reset(seed=42)` on the default board, food is at `(10, 16)` and the
actual observation is `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]`: safe immediate
moves, heading right, and food directly below.

Danger means the next move would hit a wall or body, accounting for the departing
tail. It does not predict starvation or future traps. Food comparisons use the
head coordinates, not the current direction. Diagonal food activates two flags;
food in the same row or column activates neither flag on that axis. When food is
`None` after a win, all food flags are zero. Terminal observations describe the
final valid board, even though no further actions may be taken.

### Actions

| Action | Meaning | Example when moving RIGHT |
| --- | --- | --- |
| 0 | Straight | RIGHT |
| 1 | Turn right (clockwise) | DOWN |
| 2 | Turn left (counterclockwise) | UP |

`relative_direction()` implements the conversion for all headings. Accept only
Python integers 0, 1, 2; booleans, floats, strings, and one-hot lists are rejected.
No action allows an immediate reversal.

### Rewards and episode boundaries

`RewardConfig(food=10, collision=-10, movement=0, win=10)` contains the initial
experimental rewards. Pass another `RewardConfig` to `SnakeEnv(rewards=...)`
to change them. Winning replaces the food reward on the final meal, so the
default winning step earns +10 total rather than +20.

`EnvConfig()` defaults to 20 by 20 cells, `max_steps=10000`,
`starvation_min_steps=400`, and `starvation_factor=100`. The starvation allowance
is deterministically `max(starvation_min_steps, starvation_factor * current_length)`.
The default therefore allows 400 moves without food for the initial length 3.
Eating resets the no-food counter to zero and updates the allowance using the
new body length. All limits must be positive integers.

Colliding or filling the grid gives `terminated=True` and `truncated=False`.
Otherwise reaching either the total step cap or the starvation allowance gives
`terminated=False` and `truncated=True`. Reaching a limit means `>=`, not `>`.
Terminal outcomes take precedence over limits; if both limits are reached on a
nonterminal step, the reason is `max_steps`. Truncation keeps the reward earned
on that step (normally 0, or +10 if food was eaten at the total step cap). It
does not apply the collision penalty. A truncated engine may still say
`game_status='running'`; the wrapper, not the engine, owns episode limits.

### Run and reproduce the random baseline

```powershell
# The supplied baseline files already exist; use a new filename for a rerun.
.\.venv\Scripts\python.exe -m snake_ai.evaluate --episodes 100 --seed-start 0 --agent-seed-start 10000 --output results/random_baseline_repeat.csv

# Focused tests and then all tests
.\.venv\Scripts\python.exe -m pytest tests/test_environment.py tests/test_evaluation.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluator uses environment seeds **0 through 99** and independent action
seeds **10000 through 10099**, reseeding both for each episode. `RandomAgent.act()`
samples uniformly from all three actions, including unsafe ones; it never uses
the observation. CLI options also expose the three limits; use `--help` for
details. Board size and rewards can be configured through the Python API.

The CSV stores one row per episode with both seeds, score, steps, total reward,
termination/truncation flags, and reason. The companion JSON stores exact seed
lists, environment configuration, rewards, Python version, policy description,
and aggregate results. Existing files are refused rather than overwritten.

Actual run on 12 November 2026, Python 3.12.1, default configuration:

| Metric | Measured result |
| --- | --- |
| Episodes | 100 |
| Average score | 0.14 |
| Median score | 0 |
| Maximum score | 2 |
| Minimum score | 0 |
| Average episode length | 66.49 moves |
| Collision rate | 100% |
| Truncation rate | 0% |
| Win rate | 0% |

A second run matched all 100 episode rows and the JSON manifest. These are an
initial baseline, not a definitive performance estimate. No observed truncations
or wins in this sample does not imply either is impossible.

Milestone 2 validation: **55 focused tests passed; 88 total tests passed**, including
the original 33. Coverage includes every heading/action conversion, observation
order and values, body/wall/tail danger parity with actual engine moves, food
flags, configurable rewards, winning, both truncation limits, boundary priority,
reset, reproducibility, saved results, and headless imports.

## Known limitations and later milestones

The 11 features omit body layout, distances, absolute position, and episode
counters. Different boards can look identical to the agent and require different
plans; the observation is not a complete Markov state. Immediate danger does not
expose future traps. Hiding the no-food counter also hides proximity to truncation.
The initial rewards and limits are experimental, not tuned settings. Tests use
tiny boards for rare wins; the random baseline provides no evidence of winning
on 20 by 20. Dependencies use version ranges; exact reproducibility across Python
versions is not guaranteed. The wrapper is Gymnasium-style, not a full Gymnasium
environment with spaces or registration.

The same tested engine rules continue to serve human play and headless clients.
Milestone 3 supplies tensor conversion and a network; the learning algorithm
remains future work as described below.

## Milestone 3: neural-network development

### CPU-only PyTorch setup

Use the existing Python 3.12 virtual environment. Install PyTorch from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu), following
[PyTorch's installation guidance](https://pytorch.org/get-started/locally/):

```powershell
Set-Location 'C:\Users\ilach\Desktop\VSCODE\snake-ai'
.\.venv\Scripts\python.exe -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,ml]'
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Verified locally: Python **3.12.1**, PyTorch **2.15.0+cpu**, CUDA build **None**.
`pyproject.toml` pins the public PyTorch version in the optional `ml` extra.
The CPU index selects the `+cpu` build; install it first as shown so the later
editable install reuses it. No CUDA, torchvision, torchaudio, or unrelated ML
libraries were installed. PyTorch's own required dependencies are installed
automatically. The game and random baseline still work without importing torch.

### Architecture and parameter count

`snake_ai/models/dqn.py` defines `DQNNetwork(nn.Module)`:

```text
11 observations -> Linear(11, 128) -> ReLU
                -> Linear(128, 128) -> ReLU
                -> Linear(128, 3) -> raw Q-values
```

`input_dim`, `hidden_size`, and `output_dim` are configurable, defaulting to 11,
128, and 3. Use float32 CPU tensors shaped `(11,)` for one observation or
`(batch_size, 11)` for a batch. Outputs have shape `(3,)` or `(batch_size, 3)`.
Invalid dimensions and non-floating inputs get clear errors; callers control
dtype/device conversion. Model inputs must match the model's dtype and device.

A Linear layer with n inputs and m outputs has `n * m` weights plus `m` biases.
`parameter_counts()` counts actual trainable tensors using `numel()`, rather
than hardcoding the architecture's expected total:

| Layer | Calculation | Actual trainable parameters |
| --- | --- | --- |
| Input to hidden 1 | 11 × 128 + 128 | 1,536 |
| Hidden 1 to hidden 2 | 128 × 128 + 128 | 16,512 |
| Hidden 2 to output | 128 × 3 + 3 | 387 |
| Total | Sum of layer counts | **18,435** |

The output has no Softmax. Future Q-values will estimate discounted future
reward for each action; they can be negative, exceed one, and need not sum to
one. Softmax would constrain them to probabilities and change their meaning.
The current untrained numbers have no learned reward interpretation.

### Untrained model demonstration

```powershell
.\.venv\Scripts\python.exe -m snake_ai.demos.network
```

`run_demo()` resets `SnakeEnv` with environment seed 42 and creates a network
with PyTorch seed 7. It converts the observation with `torch.tensor(...,
dtype=torch.float32)`, uses `model.eval()` and `torch.no_grad()` for inference,
selects the largest raw output with `argmax`, and sends that one action to the
environment. It prints both observations, outputs, selected action, reward,
termination flags, metadata, and parameter counts. It opens no window.

Actual output on this installation (rounded):

```text
Observation: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
Q-values: [-0.0073753223, -0.1270019710, -0.0337755084]
Selected action: 0 (straight), absolute direction RIGHT
Reward: 0; terminated: false; truncated: false; game status: running
```

This is an **untrained network demonstration**. The choice comes from random
initial weights and is not evidence of intelligent or learned play. Argmax
simply selects the largest number, even if all three are negative.

### One educational optimisation step

```powershell
.\.venv\Scripts\python.exe -m snake_ai.demos.optimisation
```

This independent example uses three fixed observation-like rows and three
invented target triples. Targets are arbitrary teaching numbers, not rewards,
Bellman targets, or claims about good actions. It creates a fresh seeded model,
performs a forward pass, computes mean squared error, clears gradients, calls
`backward()`, and runs **one** SGD step at learning rate **0.001**. A new forward
pass measures the updated loss on the same batch. It saves no model or files.

Actual results:

| Quantity | Measured value |
| --- | --- |
| Loss before | 0.5388405919 |
| Loss after | 0.5384227037 |
| Gradient L2 norm | 0.6465501032 |
| Parameter tensors changed | 6 |
| Optimiser steps | 1 |

The loss decreased for this example. One optimiser step does **not** guarantee a
lower loss for every configuration, and lower error on these invented targets
does not mean better Snake play. Both demos reseed PyTorch's global RNG when
called; the model constructor itself does not set seeds.

### Verification and limitations

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dqn.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

On 12 November 2026: **22 focused tests passed**; **110 full-suite tests passed**,
including all 88 earlier tests. Dependency checks passed. Tests verify single
and batch shapes, finite outputs, input validation, configurable architecture,
raw unrestricted output values, parameter counts, finite nonzero gradients,
changed weights, same-seed initialisation/predictions, actual environment input,
repeatable demos, and headless execution. No graphical manual play-through was
performed by Codex for this milestone.

There is one focused-test warning and two full-suite warnings from dependencies:
PyTorch cannot initialise its optional NumPy bridge because NumPy is not installed;
Pygame also reports deprecated `pkg_resources` use after its dependency update.
Tensor operations and both demos work without NumPy; no warning is suppressed.
The pure game/environment tests do not import torch. Running the model tests
requires the `ml` extra. Seeds reproduce results on this same installation, but
identical values across versions, devices, or platforms are not guaranteed.

## Before Milestone 4

The network has never learned to play Snake. Before approved DQN work, revisit
the partial observation and reward choices, specify handling of truncations in
learning targets, then implement the learning algorithm, replay memory,
exploration, target network, training/checkpoint workflow, and evaluation on
separate seeds. None of that machinery is part of Milestone 3. Existing random
baseline files remain unchanged and no new performance comparison is claimed.
