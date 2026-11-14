# SnakeAI

SnakeAI is an educational reinforcement-learning project with a playable Snake
game, a headless environment, and a Deep Q-Network built with PyTorch. A trained
model can play through a Pygame interface that displays its action-value estimates.
Game rules, rendering, model code, and training are separate so the same rules
serve human play, automated tests, training, and inference.

The initial CPU experiment trained one seed for 500 episodes. On 100 held-out
games, greedy DQN averaged **34.81** food items versus **0.14** for a fresh random
policy using the same environment seeds. Every episode still ended in collision.
This is a single-training-seed result, not evidence of performance across multiple
training runs. Per-episode data and configurations are tracked under `results/`.

## Technology stack

- Python 3.11+; verified locally with Python 3.12.1 on Windows.
- Pygame for the game window and keyboard controls.
- PyTorch for the 11 → 128 → 128 → 3 network, autograd, and Adam optimisation.
- Matplotlib for training charts; pytest for automated tests.
- Git and a private GitHub repository for source and experiment records.

## Six-milestone roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | Playable Snake and a separate testable game engine | Complete |
| 2 | Headless RL environment, 11 features, relative actions, random baseline | Complete |
| 3 | PyTorch network and educational inference/optimisation examples | Complete |
| 4 | Replay, standard DQN training, checkpoints, and paired held-out evaluation | Complete: one bounded training seed |
| 5 | Agent inspection and further analysis | 5A playback, 5B/5C diagnostics, and 5D feature comparison complete |
| 6 | Portfolio presentation and project packaging | Planned; not implemented |

The remaining roadmap is provisional. Further analysis, algorithm changes,
additional experiments, and portfolio features have not been completed.

## Clone and install

The repository is private, so cloning requires access and GitHub authentication.
For a new checkout, run the following in PowerShell from your chosen parent folder:

```powershell
git clone https://github.com/Chaytou-141/snake-ai.git
Set-Location snake-ai
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch==2.15.0' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,training]'
.\.venv\Scripts\python.exe -m pytest -q
```

Installing from the CPU index first ensures that the editable install reuses the
CPU PyTorch build. The `training` extra includes plotting dependencies; `dev`
includes pytest. Human play alone needs only `pip install -e .`.

**Cloning does not include trained weights.** Large generated `.pt`/`.pth`
checkpoints are ignored to keep Git history focused on source and small useful
experiment records. The verified model remains on the original machine at
`results/milestone4_initial/checkpoints/final.pt`. It is not uploaded to GitHub,
Git LFS, releases, or any other service. Playback on another machine requires an
existing compatible local checkpoint or an explicitly chosen new training run.
A new run may not reproduce the exact measured results. Its checkpoint path can
be passed with `--checkpoint`; training never happens automatically at launch.

Quick commands after setup:

```powershell
# Human play and automated tests work without the saved trained checkpoint.
.\.venv\Scripts\python.exe -m snake_ai
.\.venv\Scripts\python.exe -m pytest -q

# Playback requires the existing local checkpoint at this path.
.\.venv\Scripts\python.exe -m snake_ai.demos.trained_ai --checkpoint results/milestone4_initial/checkpoints/final.pt

# Optional new bounded training run; do not overwrite recorded experiments.
.\.venv\Scripts\python.exe -m snake_ai.training.train --config configs/dqn.json --output results/training_new
```

## VS Code

Open the `snake-ai` folder and install the Microsoft Python and Python Debugger
extensions for editor testing and F5 launch. Project settings select `.venv` and
enable pytest discovery. If VS Code retains another interpreter, use **Python:
Select Interpreter** and choose `.venv\Scripts\python.exe`. The direct interpreter
commands above avoid PowerShell activation-policy issues.

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
    evaluation/
      __init__.py
      collision_diagnostics.py
      trap_analysis.py
    models/
      __init__.py
      dqn.py
    demos/
      __init__.py
      network.py
      optimisation.py
      trained_ai.py
    agents/
      __init__.py
      dqn_agent.py
    training/
      __init__.py
      config.py
      replay_buffer.py
      train.py
      evaluate.py
      charts.py
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
    test_training.py
    test_trained_ai.py
    test_collision_diagnostics.py
    test_trap_analysis.py
  configs/
    dqn.json
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
`74211f1` before Milestone 2 edits. Milestone 2 was reverified with 88 passing
tests and committed as `fccb88f` (`feat: complete SnakeAI reinforcement learning environment`)
before Milestone 3 work. Existing baseline and repeat results were preserved and
verified identical. Milestone 3 was reverified with 110 passing tests and committed
as `18c462c06d37200f50d91d1fd762842717adc2d2` with message
`feat: complete SnakeAI PyTorch neural network`. Milestone 4 was reverified with
146 passing tests, its checkpoint and recorded results were checked, and it was
committed as `e0bd9cf7078ee886bf762019a94365f89d4eb575` with message
`feat: implement and train SnakeAI DQN agent`. Milestone 5A is verified with
173 passing tests and committed as `1babc0231f249d8400447ec3f7d74cd765bd4d63`
(`feat: add trained SnakeAI playback visualisation`). These are the published
hashes after the authorised commit-email rewrite; contents and original dates
were preserved. Generated checkpoints remain local and ignored. Milestone 5B was
verified with 190 passing tests, checked against the original 100-episode results,
and committed as `48709f81cbb2a970a7f4d389354d5b90d3514fcb`
(`feat: add SnakeAI collision diagnostics`). Milestone 5C is verified with 208
passing tests for the checkpoint `feat: add SnakeAI trap formation analysis`.

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
.\.venv\Scripts\python.exe -m snake_ai.evaluate --episodes 100 --seed-start 0 --agent-seed-start 10000 --output results/random_baseline_new.csv

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
Milestone 3 supplies tensor conversion and a network; Milestone 4 implements
the standard DQN learning algorithm described below.

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

At the Milestone 3 checkpoint there was one focused-test warning and two
full-suite warnings: NumPy was absent and Pygame used deprecated `pkg_resources`.
Milestone 4 subsequently installed NumPy with Matplotlib, resolving the bridge
warning; the Pygame deprecation remains. Neither warning is suppressed.
The pure game/environment tests do not import torch. Running the model tests
requires the `ml` extra. Seeds reproduce results on this same installation, but
identical values across versions, devices, or platforms are not guaranteed.

The two Milestone 3 educational demos remain untrained examples. The trained
agent and its separate measured results are described in Milestone 4 below.

## Milestone 4: standard DQN training

### Setup and workflow

```powershell
# Keep the CPU PyTorch installation; add plotting dependencies.
.\.venv\Scripts\python.exe -m pip install -e '.[dev,training]'

# Stage A: focused unit tests, then the full suite
.\.venv\Scripts\python.exe -m pytest tests/test_training.py -q
.\.venv\Scripts\python.exe -m pytest -q

# Stage B: smoke test. Choose new output names; existing directories are refused.
.\.venv\Scripts\python.exe -m snake_ai.training.train --output results/smoke_new --episodes 10 --max-steps 500 --learning-starts 64 --batch-size 16

# Stage C: bounded experiment, maximum 500 episodes OR 100,000 moves
.\.venv\Scripts\python.exe -m snake_ai.training.train --config configs/dqn.json --output results/training_new

# Evaluate the new final model on the reserved seeds with a fresh random policy
.\.venv\Scripts\python.exe -m snake_ai.training.evaluate --checkpoint results/training_new/checkpoints/final.pt --output results/evaluation_new
```

CPU-only PyTorch remains **2.15.0+cpu**. Plotting uses Matplotlib **3.11.2**, which
installed NumPy **2.5.4** as a required dependency. This resolves the previous
optional PyTorch/NumPy warning. The Pygame `pkg_resources` deprecation remains.
No cloud service, GPU dependency, dashboard, or RL framework was introduced.

`configs/dqn.json` is the experiment's central configuration; `TrainingConfig`
loads and validates it. Saved run `config.json` files capture the effective
configuration, Python/PyTorch versions, device, and reserved evaluation seeds.
Rewards, board size, and episode limits are unchanged from Milestone 2.

| Hyperparameter | Initial setting |
| --- | --- |
| Network | 11 → 128 → ReLU → 128 → ReLU → 3 raw values |
| Optimiser / learning rate | Adam / 0.0005 |
| Gamma | 0.99 |
| Replay capacity / batch size | 50,000 / 64 |
| Learning starts | At 1,000 collected environment steps |
| Target synchronisation | Every 1,000 optimiser updates |
| Gradient clipping | Combined gradient norm at most 10 |
| Epsilon | 1.0 to 0.05, linear over 50,000 environment steps |
| Budget | 500 episodes, at most 100,000 environment steps |
| Model/exploration seed | 7; replay RNG 8; action RNG 9 |
| Training food seeds | 100000 + episode-start index (zero-based) |
| CPU threads | 1 |

These settings are experimental starting points. There was no hyperparameter
search and the budget was not extended. Environment moves count interactions;
optimiser updates count gradient steps and start only after warmup. The agent
attempts one update per collected move when both warmup and batch size allow.

### Replay, exploration, and the DQN update

`ReplayBuffer` keeps a bounded deque of copied experiences:
`(state, action, reward, next_state, terminated, truncated)`. Oldest experiences
leave when capacity is reached. Sampling uses a dedicated RNG, without
replacement within a batch. State tensors are float32, actions int64, and flags
boolean. Invalid dimensions, nonfinite values, invalid actions, and insufficient
samples produce errors. Mixing past moves reduces the correlation found in
consecutive gameplay and lets the same experience be reused.

Epsilon-greedy chooses a random action with probability epsilon, otherwise the
action with highest online Q-value. At n collected moves,
`epsilon = 1.0 + min(n / 50000, 1) * (0.05 - 1.0)`. The schedule advances when
experience is stored, not when predictions or gradient updates run. Greedy
evaluation uses epsilon 0 without changing the stored schedule or action RNG.

The online network predicts `Q(state, selected_action)`, extracted with `gather`.
The frozen target network supplies the next-state maximum:

```text
target = reward + gamma * (1 - terminated) * max_a Q_target(next_state, a)
loss = SmoothL1(Q_online(state, selected_action), target)
```

Target calculations run inside `torch.no_grad()`; target parameters also have
gradients disabled. Only the online network receives Adam updates. Every 1,000
optimiser updates, its parameters are copied to the target network. This is
standard DQN, not Double DQN. Gradients are clipped before the optimiser step;
the update CSV records the norm before clipping. Nonfinite loss/gradient errors
stop the run and preserve a final checkpoint.

Genuine collision or win zeroes the future term. Ordinary time limits and the
starvation truncation retain bootstrapping, as does a run-level step cap. The
stored next observation is the final pre-reset observation, never the next
episode's initial board. Rewards are unchanged: food +10, collision -10,
movement 0, win +10 replacing the final food reward.

### Files, interruptions, and loading

Each run has `episodes.csv`, `updates.csv`, `config.json`, `summary.json`, a
`training_charts.png`, and a `checkpoints/` directory. The episode CSV records
score, total reward, length, epsilon, mean loss (blank before any updates), end
reason, both counters, food seed, and whether the episode completed. The update
CSV records every loss, pre-clip gradient norm, epsilon, and both counters.

Charts show score per completed episode, rolling mean score (up to 25 episodes),
episode reward, loss per optimiser update, and epsilon versus environment steps.
A run cap or interrupt can leave a partial episode; that row is labelled and
excluded from completed-episode score statistics. Ctrl+C saves current progress,
flushed CSVs, and `final.pt`. Periodic checkpoints are saved every 100 episode
rows. Each run directory is new, and checkpoint saves refuse existing paths.
Checkpoints are written to a temporary file before atomic replacement into the
new destination. `.pt`, `.pth`, and temporary files are ignored by Git.

```python
from pathlib import Path
from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import SnakeEnv

agent = DQNAgent.load(Path("results/milestone4_initial/checkpoints/final.pt"))
env = SnakeEnv(agent.config.environment, agent.config.rewards)
observation, _ = env.reset(seed=800000)  # A demonstration seed, not training.
action = agent.act(observation, evaluate=True)
print(agent.predict(observation), action)
```

Checkpoints contain online/target state, Adam state, counters, hyperparameters,
seed settings, training food seeds, exploration progress, action RNG state, and
torch RNG state. Default loading is for inference: online gradients are disabled
and `act(..., evaluate=True)` is greedy. For continued training:

```powershell
.\.venv\Scripts\python.exe -m snake_ai.training.train --config configs/dqn.json --resume-approx results/milestone4_initial/checkpoints/final.pt --output results/continued_new
```

**Resumption is approximate.** Replay memory, replay sampling RNG progress, and
the active environment are not restored. Adam, counters, target network, action
RNG, torch RNG, and epsilon progress are restored; replay starts empty and must
refill to batch size. A fresh episode uses the next training seed, abandoning an
unfinished episode. The checkpoint's original configuration must match; budgets
apply to additional episodes/moves in the new run. This cannot reproduce an
uninterrupted run exactly. Do not use inference-mode loading to resume training.

### Recorded experiment and evaluation

The initial experiment used the untouched defaults above and training seeds
100000–100499. Actual training completed **500 episodes in 54.23 seconds**, with
**33,629 moves** and **32,630 optimiser updates**. The duration covers collection,
updates, and periodic checkpoints, excluding final chart generation. Final
epsilon was **0.361049**, since the episode budget ended before 50,000 moves.
Completed training scores: mean **1.05**, median **1**, minimum **0**, maximum
**8**; the last 100 episodes averaged **1.96**. Training includes exploration;
these scores are not greedy evaluation results.

The smoke run used a separate directory and a 500-move cap: 500 experiences,
437 finite updates, changed weights, successful checkpoint loading, and a check
that greedy evaluation does not change weights, epsilon, counters, or action RNG.
It completed five episodes and logged one partial episode at the cap.

Saved evidence:

- `results/milestone4_smoke/`: smoke metrics, configuration, checkpoint, chart.
- `results/milestone4_initial/`: training CSVs, summary, configuration, chart.
- `results/milestone4_initial/checkpoints/final.pt`: final trained model.
- `results/milestone4_initial/checkpoints/episode-000100.pt` through
  `episode-000500.pt`: periodic checkpoints.
- `results/milestone4_evaluation/`: paired held-out episode results and summary.

Evaluation uses environment seeds **900000–900099** for both policies and random
policy seeds **2000000–2000099**. They are disjoint from training. DQN uses greedy
actions with no exploration; the evaluator verifies that online/target weights,
training counters, epsilon, and action RNG stay unchanged. The saved summary
records the checkpoint SHA-256, complete configuration, exact seed lists, both
policies' metrics, and DQN-minus-random differences. This is a new paired
comparison; the earlier 0.14-score baseline used different seeds and is preserved.

Measured held-out results from the final checkpoint:

| Metric | Greedy DQN | Fresh random | DQN minus random |
| --- | --- | --- | --- |
| Episodes | 100 | 100 | — |
| Average score | 34.81 | 0.14 | +34.67 |
| Median score | 37 | 0 | +37 |
| Minimum score | 8 | 0 | +8 |
| Maximum score | 58 | 1 | +57 |
| Average episode length | 637.28 | 59.64 | +577.64 |
| Collision rate | 100% | 100% | 0 percentage points |
| Truncation rate | 0% | 0% | 0 percentage points |
| Win rate | 0% | 0% | 0 percentage points |

The trained agent demonstrates measurable score improvement in this paired
sample, while still eventually colliding in every evaluated episode. The fresh
random mean happens to match the earlier baseline's 0.14; the seed sets and
maximum scores differ, so these remain separate experiments. Training scores
are much lower than greedy evaluation because the bounded run retained substantial
random exploration (final epsilon 0.361049).

Final validation: **36 focused training tests passed; 146 full-suite tests
passed**, including all 110 earlier tests. The sole warning is the existing
Pygame packaging deprecation. `pip check` passes, and NumPy-to-tensor integration
was checked successfully. Tests cover replay capacity/copying/dtypes, seeded
sampling, action gathering, detached terminal and truncated targets, exploration,
gradient clipping, hard target copies, checkpoint loading, approximate resume,
pipeline reproducibility, cap handling, interrupt recovery, held-out separation,
and headless chart generation. No manual graphical play-through was performed.

This single training seed and 100 evaluation episodes support only an initial
comparison. Lower loss is not proof of better play. The observation still hides
body geometry, distances, and remaining budgets. Bootstrapping across starvation
is an explicit modelling choice for an externally imposed budget. No Milestone
5 improvements, additional training budgets, or tuning against held-out results
are part of this milestone.

The implementation uses the standard concepts described in the
[official PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
with this project's existing environment, linear exploration schedule, and hard
target synchronisation. Our measured results come from the saved local CSVs,
not from the tutorial.

## Milestone 5A: watch the trained agent

Run from the project folder using the existing virtual environment:

```powershell
Set-Location 'C:\Users\ilach\Desktop\VSCODE\snake-ai'
.\.venv\Scripts\python.exe -m snake_ai.demos.trained_ai

# Explicit checkpoint, food seed, and speed
.\.venv\Scripts\python.exe -m snake_ai.demos.trained_ai --checkpoint results/milestone4_initial/checkpoints/final.pt --seed 800000 --speed 10
```

The default checkpoint is `results/milestone4_initial/checkpoints/final.pt`.
`--checkpoint` accepts another checkpoint saved by this project's DQN agent.
The file must already exist locally; model files are deliberately ignored by
Git. No retraining, downloading, or checkpoint creation happens when playback
starts. The existing Pygame and PyTorch installation supplies all dependencies.

### What is displayed

The game area reuses the original Pygame renderer: grid, food, snake, brighter
head, and score. A separate panel shows the checkpoint filename, playback state
and speed, episode/seed, score, moves, current direction, latest reward, and total
episode reward. The three Q-values are labelled **Straight**, **Right**, and
**Left**, with the selected argmax highlighted.

During play and pause, **NEXT ACTION Q-VALUES** describe the currently displayed
board and the move the model will take next. They refresh after each successful
environment step. At a win, collision, or truncation, the panel instead says
**FINAL MOVE Q-VALUES** and **Values before final move**, retaining the values
that selected the last executed action. No new action is executed after an
episode ends. The final board, score, reward, and end reason remain visible.

Q-values estimate future discounted reward under the learned policy. They are
raw numerical estimates, not probabilities or a guarantee of success. The panel
does not convert them to percentages or apply Softmax.

### Controls

| Key | Action |
| --- | --- |
| Space | Pause or resume; ended episodes remain stopped |
| R | Retry the current episode using the same seed; resume playback |
| N | Start the next episode with seed increased by one |
| + | Increase playback speed by one move per second |
| - | Decrease playback speed by one move per second |
| Escape / window close | Exit |

The main keyboard's `=` key also increases speed, making the usual Shift-plus
key work. Numeric keypad plus/minus are supported. Speed defaults to 10 moves
per second and is limited to 1–60; it changes timing only. Arrow keys do not
steer the snake in AI playback. Pausing, retrying, or changing speed clears the
movement timer so there is no accumulated burst of moves afterward.

The initial seed defaults to 800000. **R** keeps the current episode number and
seed, allowing a repeat of the same trajectory. **N** increases the episode
number and seed, whether an episode is running or ended. Both reset score,
reward, moves, and end flags while retaining playback speed. There is no
automatic episode restart, giving time to inspect a failure.

### Loading and environment compatibility

`PlaybackSession` calls `DQNAgent.load()` for inference, which loads checkpoint
tensors on CPU and strictly checks them against the model configuration. The
demo uses `model.eval()`, disables parameter gradients, and runs each forward
pass under `torch.inference_mode()`. It checks that the model accepts 11 features
and returns three finite values. It then selects `argmax` directly, without
using epsilon-greedy exploration or advancing training counters.

The session creates `SnakeEnv` from the checkpoint's stored environment and
reward configuration. It calls the existing `reset()` and `step()` methods;
there is no second observation encoder, action conversion, or collision rule.
Score, moves, rewards, and termination/truncation flags come from the environment.
The CLI does not override board dimensions, rewards, or episode limits. A
checkpoint with incompatible weight shapes or invalid outputs fails explicitly.
The interface is designed primarily for the verified 20 by 20 checkpoint.

The renderer gained only optional control-text and terminal-overlay arguments,
whose defaults preserve its human-play display. Human play still launches with
`.\.venv\Scripts\python.exe -m snake_ai` and retains its original controls.

### Verification and limitations

Milestone 5A validation: **27 focused tests passed; 173 full-suite tests passed**.
The existing Pygame packaging deprecation is the only warning. Tests cover
checkpoint loading and architecture, recorded environment settings, observation
size, argmax and relative actions, inference mode and frozen weights, rewards,
retry/new-episode behavior, pause, win/loss/truncation handling, speed limits,
headless rendering, keyboard controls, and clean exit. Run them with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_trained_ai.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The actual trained checkpoint was also run for 80 headless playback moves on
seed 800000, collecting seven food items. Its rendered 940 by 700 frame was
visually inspected, and weight equality and the checkpoint hash were checked.
This was an automated headless rendering check, not a manual live-window
keyboard play-through or a new performance evaluation.

The saved model's architecture remains 11 → 128 → 128 → 3 with 18,435 parameters.
Its SHA-256 remains
`ac2ba41422d265ac4a0384126cfa42de7c6c7bd74f468bc703901af96cade726`.
The existing trained checkpoints and all saved result files are preserved.

The agent can still crash: its 11 features omit full body geometry and long-term
traps, its value estimates are approximate, and its original training budget was
limited. All 100 original held-out DQN episodes eventually collided despite
improved scores. Watching playback does not improve the model, and pausing or
changing speed does not change its decisions. A chosen seed on another library
version may not give exactly identical predictions. No Milestone 5C work,
retraining, architecture changes, reward changes, or new dependencies are included.

## Milestone 5B: collision diagnostics

Diagnostics replay the saved greedy policy headlessly on the original 100
evaluation seeds (900000–900099). Before publishing analysis, the workflow
checks the checkpoint hash and configuration and reproduces **every episode's
score, moves, reward, end flags/reason and all aggregate metrics** from Milestone 4.
It never trains or changes the policy, environment, rewards, or visual playback.

```powershell
.\.venv\Scripts\python.exe -m snake_ai.evaluation.collision_diagnostics

# The saved output directory already exists; choose a fresh name for a repeat.
.\.venv\Scripts\python.exe -m snake_ai.evaluation.collision_diagnostics --output results/milestone5b_repeat
```

The command requires the existing local checkpoint and Milestone 4 reference
files. `--checkpoint` and `--reference` can point to matching copies. Output
directories are never overwritten; a reproduction mismatch stops the analysis.

The three final-event categories are:

- **A — avoidable immediate collision:** another action was safe for the next move.
- **B — no immediately safe action:** all three actions would collide immediately.
- **C — other or uncertain:** another end reason, or inconsistent evidence.

Safety uses the engine's `would_collide()` and existing relative-action mapping,
including the departing tail rule. A safe next move does not prove future
survival. Type B does not prove that earlier decisions could not avoid the trap.
Earlier movements are not automatically labelled mistakes.

Verified results: **100 episodes**, mean score **34.81**, median **37**, maximum
**58**, mean moves **637.28**, collision rate **100%**, matching the original
evaluation exactly. Categories: **A: 0 (0%)**, **B: 100 (100%)**, **C: 0 (0%)**.
All final observations exposed danger in all three directions. There was no
immediately safe final alternative, so selected-versus-safe Q-value gaps are
undefined, not zero. Selected final Q-values averaged **-9.417852**, ranging from
**-10.043816 to -8.523903**. Their sign is not the safety classifier.

Evidence is saved under `results/milestone5b_collision_diagnostics/`:

- [Analysis report](results/milestone5b_collision_diagnostics/report.md), including schema, examples, and limitations.
- `summary.json`: configuration, checkpoint/reference hashes, exact seeds, reproduction checks, and statistics.
- `episodes.csv`: one row per episode; `collisions.csv`: one row per final collision.
- `final_moves.csv`: 3,000 retained moves, at most 30 per episode, with actual
  pre/post head, complete bodies, food, score, observations, Q-values, action,
  safety flags, reward, and end flags/reason. Structured cells contain JSON.
- `example_failure.json`: detailed first Type B and maximum-score examples.
  The Type A example is `null` because none occurred; no example is invented.
- `collision_categories.png`: a static count chart.

The engine keeps its last valid head/body on a collision. Logs preserve that
actual post-state and separately label the projected `attempted_head`.
Q-values estimate discounted future reward, not probabilities. All 23 earlier
checkpoint/result files remain byte-for-byte unchanged. No new dependencies
were added, and diagnostics did not modify model weights or training state.

Validation: **17 focused tests passed; 190 full-suite tests passed**, with only
the existing Pygame packaging warning. Run focused checks with
`.\.venv\Scripts\python.exe -m pytest tests/test_collision_diagnostics.py -q`.
The retained histories cover only 30 moves and the policy sees only 11 features;
these observations do not establish a particular cause of earlier trapping.
The next investigation should examine earlier planning decisions under a
predefined analysis, with future model comparisons using fresh evaluation seeds
and multiple training seeds. No improvements are implemented in this milestone.

## Milestone 5C: trap formation and future safety analysis

This read-only analysis measures **every pre-move state**, rather than only the
last 30 moves. It reuses the collision diagnostics' actual greedy moves and the
engine's immediate-collision predictor. The policy still receives its original
11 features; diagnostic geometry is never used to select actions.

```powershell
.\.venv\Scripts\python.exe -m snake_ai.evaluation.trap_analysis

# Existing output is preserved; use a new directory for a repeat.
.\.venv\Scripts\python.exe -m snake_ai.evaluation.trap_analysis --output results/milestone5c_repeat
```

The command requires the local trained checkpoint and the saved Milestone 4 and
5B reference files. It checks configuration, hashes, exact seeds, per-episode
outcomes, and aggregate metrics against both milestones. An `INCOMPLETE` marker
remains if a run fails or reproduction does not pass; no verified analysis report
is issued for a mismatch. Earlier results are never overwritten.

### What the geometry means

- **Immediate options:** 0–3 safe relative actions using the engine's existing
  tail-vacating rule. Logs identify the first exact two-, one-, and zero-safe
  states, transitions that reduce options, and recovery from one-safe runs.
- **Obstacle distance:** the number of free cells on the straight/right/left
  ray before the first wall or current body cell. The tail counts as a static
  obstacle here, so distance zero can coexist with a legal tail-vacating move.
- **Candidate reachable space:** after a legal first move, BFS counts connected
  four-neighbour cells, including the new head. The tail leaves unless known
  food is eaten; then the resulting body is frozen for flood fill. No future
  food is sampled. A collision candidate has `null` space rather than zero.

Static space is not a dynamic survival prediction. It ignores subsequent tail
movement, future growth, and movement-sequence constraints. A larger region is
not always a better action. The optional three-move lookahead was not implemented;
even that horizon could miss a deeper trap or encounter unknown future food.

### Measured results

All **100 episodes** on seeds 900000–900099 reproduced Milestone 4 and 5B exactly:
mean score **34.81**, median **37**, maximum **58**, mean moves **637.28**, and
100 collisions. The full trace contains **63,728 moves** and averages **2.551469**
immediately safe actions per move.

| Safe actions before move | Count | Percentage |
| --- | ---: | ---: |
| 0 | 100 | 0.157% |
| 1 | 3,126 | 4.905% |
| 2 | 22,032 | 34.572% |
| 3 | 38,470 | 60.366% |

There were **1,704 consecutive one-safe runs**: **1,628 recovered** to at least
two options (**95.54%**), while 76 transitioned to zero options. At the individual
visit level, 1,628 of 3,126 one-safe states recovered on their very next step
(52.08%). The different denominators answer different questions. The selected
move reduced next-state options on **5,094 of 63,628** nonterminal transitions
(**8.01%**); this does not label those decisions mistakes.

Mean static obstacle distances were **7.960 straight**, **7.468 right**, and
**6.993 left** free cells. Using thresholds fixed before the run, selected space
was **at most 20 cells** on **819 legal moves (1.287%)**, entered on 129 occasions.
The selected action had **at least 20 fewer reachable cells** than the best safe
candidate on **184 moves (0.289%)**. Pooled correlation between selected static
space and remaining moves was **r = 0.284**. This is descriptive: adjacent states
are dependent, longer episodes have more weight, and body length/episode phase
can confound the association. It is not evidence that more space causes survival.

In the last five moves, mean immediate options were **1.186**, compared with
**2.551** overall. On every last legal move, selected static space was one cell;
24 episodes had a larger-space alternative then, including 23 with a gap of at
least 20 cells. These alternatives were not played out, so later survival is
unknown. The decline describes an aggregate pattern, not an irreversible trap
criterion for every individual state.

Three predefined example roles illustrate the limits:

- **Seed 900034**, score 58: at move 1360 the agent chose right with 12 reachable
  cells versus 328 for left; collision followed 12 moves later.
- **Seed 900045**, score 58: at move 152 it chose right with one static cell
  versus 384 for straight/left, yet survived **1,239 more moves**. The moving tail
  immediately provided a legal continuation. Area one is not guaranteed death.
- **Seed 900073**, score 57: at move 476 it chose left with 29 cells versus 342
  for right, then continued **818 moves**. Across this episode, 61 one-safe runs
  recovered before the eventual collision.

### Saved evidence and tests

Files are under `results/milestone5c_trap_analysis/`:

- [Report and schema](results/milestone5c_trap_analysis/report.md) and `summary.json`.
- `episodes.csv`: outcomes, first safe-count occurrences, recoveries, option
  reductions, low-space visits/entries, and substantial-deficit counts.
- `trap_events.csv`: selected event rows without duplicating full bodies.
- `state_traces.jsonl.gz`: streamed compressed JSON, one pre-state body per
  move, original observations/Q-values/actions, geometry, reward/end information,
  and actual post head/food/score. Null denotes an inapplicable candidate or
  terminal post-safe count. Array action order is straight/right/left.
- [Safe actions](results/milestone5c_trap_analysis/safe_actions.png),
  [reachable space](results/milestone5c_trap_analysis/reachable_space.png), and
  [action comparison](results/milestone5c_trap_analysis/action_comparison.png)
  charts, labelled with actual example seeds and episodes.

**18 focused tests and 208 total tests pass**, with the existing Pygame packaging
warning. Run focused tests with
`.\.venv\Scripts\python.exe -m pytest tests/test_trap_analysis.py -q`.
All 30 older saved files and the model remain unchanged. No dependencies,
training, agent, observation, reward, architecture, or playback changes were made.

The next controlled experiment should predefine a baseline versus a
geometry-informed observation hypothesis, matched training budgets, independent
training seeds, and fresh untouched evaluation seeds. This analysis motivates
testing that hypothesis; it does not establish that adding features will help.

## Milestone 5D: controlled observation comparison

Research question: **does adding three static obstacle-distance features improve
mean food score over the original 11-feature observation?** The hypothesis and
protocol were fixed before training in `configs/feature_comparison_v1.json`.
Model B is not assumed to be better. This experiment uses six fresh models,
not the earlier seed-7 checkpoint.

### Observation modes and compatibility

`SnakeEnv(..., observation_mode='original11')` remains the default. Model A uses
the original 11 binary features and **11 → 128 → 128 → 3**, with **18,435** parameters.
`observation_mode='distance14'` appends straight/right/left distances in that order.
Model B uses **14 → 128 → 128 → 3**, with **18,819** parameters.

The added values count consecutive free cells before the current wall/body,
including the tail as an occupied obstacle, divided by
`max(board_width, board_height) - 1` (**19** for this 20×20 board). Inputs stay in
[0,1]. This is the exact static distance definition from 5C, moved to a small
standard-library-only geometry helper so the environment does not import ML
or plotting libraries. Immediate danger still respects a departing tail; the
static distance can be zero even when moving into that cell is legal.

The first 11 values do not change. `TrainingConfig` derives input/replay sizes
from the selected mode; checkpoint configuration preserves that choice, and
training/evaluation construct matching encoders. Old checkpoints without a mode
field default to 11 features, with the legacy configuration schema preserved.
The original playback and 5B/5C reference workflows remain for the original
11-feature checkpoint; the comparison runner evaluates both new modes.

### Fixed training and evaluation protocol

Each variant trains from scratch with independent training seeds **11, 22, 33**.
Run order is A11, B11, A22, B22, A33, B33. Each run is limited to **500 episodes
or 100,000 moves**, whichever comes first. Shared settings are Adam 0.0005,
gamma 0.99, replay 50,000, batch 64, learning starts 1,000, target copy every
1,000 optimiser updates, gradient norm cap 10, epsilon 1.0 to 0.05 over 50,000
environment steps, and two 128-neuron hidden layers. Rewards, board, starvation,
and episode limits are unchanged. CPU uses one thread and deterministic
algorithms where supported.

For training seed s, Python and PyTorch use s; replay uses s+1 and exploration
s+2. Both variants reset episode index i with food seed `200000 + 1000*s + i`.
Every final model is greedily evaluated on the same **100 fresh environment
seeds 910000–910099**, disjoint from training, development, and earlier evaluation
seeds. No best-checkpoint selection, best-seed selection, or hyperparameter
search is performed. These seeds are used only after training each model.

Equal episode budgets need not give equal numbers of experiences or updates;
actual counts are recorded. Equal food seeds provide paired initial conditions,
not identical trajectories. The different input size changes random weight
initialisation draws and adds 384 parameters, so hidden tensors need not match.
Reproducibility is checked on this installation, not promised across platforms
or library versions.

### Commands and saved outputs

Run focused validation first, then the separate development smoke test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_feature_comparison.py -q
.\.venv\Scripts\python.exe -m snake_ai.experiments.feature_comparison --smoke
.\.venv\Scripts\python.exe -m snake_ai.experiments.feature_comparison

# Complete or inspect an individual pair without repeating successful others.
.\.venv\Scripts\python.exe -m snake_ai.experiments.feature_comparison --model A --seed 11
.\.venv\Scripts\python.exe -m snake_ai.experiments.feature_comparison --aggregate-only
```

Smoke testing uses development seed **77**, evaluation seeds **920000–920004**,
and temporary training directories. Both modes must change weights, make finite
updates, load checkpoints, and evaluate with frozen weights before full runs are
allowed. Smoke scores are not scientific findings. Its technical checks are
recorded in `validation.json`.

Outputs live under `results/milestone5d_feature_comparison/`. `protocol.json` and
`metadata.json` freeze configuration, software versions, and relevant source
hashes. Each `runs/<model>_seed<seed>/training/` contains metrics and ignored
checkpoints. Each run saves `evaluation.csv`, `evaluation.json`, and a hashed
`complete.json` marker. Completed runs are verified and reused, never retrained.
An already completed training stage can be reused to finish evaluation. Partial
training or ambiguous partial evaluation artifacts stop for review instead of
being overwritten or silently retried. Changes to protocol/source/software
require review and a new protocol version/directory before affected reruns.

Root `per_seed_summary.csv`, `comparison_summary.json`, and `report.md` present
all six runs. Four charts show per-seed mean score, rolling training performance,
episode length, and between-training-seed variability. The error bars are sample
standard deviations across three training-seed means, not confidence intervals.

The statistical unit is an independent training run. Report the mean and sample
SD of three seed means for each variant, three paired B-minus-A differences, and
their mean. The 300 evaluation episodes per variant are not 300 independent
training replications. Conclusions remain preliminary; no unnecessary
significance tests are used. See the saved report for the actual completed-run
measurements and their limitations.

### Actual 5D results

All six runs completed **500 episodes** without reaching the 100,000-step cap.
Each final checkpoint was evaluated once on all 100 specified evaluation seeds.

| Model | Training seed | Training moves | Optimiser updates | Evaluation mean score |
| --- | ---: | ---: | ---: | ---: |
| A: 11 features | 11 | 33,282 | 32,283 | 29.70 |
| B: 14 features | 11 | 30,903 | 29,904 | 22.91 |
| A: 11 features | 22 | 35,500 | 34,501 | 31.51 |
| B: 14 features | 22 | 34,005 | 33,006 | 13.31 |
| A: 11 features | 33 | 33,742 | 32,743 | 25.94 |
| B: 14 features | 33 | 30,170 | 29,171 | 23.67 |

The mean of A's three training-seed means was **29.05**, with sample SD **2.84**.
B's was **19.96**, with sample SD **5.77**. Paired B−A differences for seeds
11, 22, and 33 were **−6.79, −18.20, and −2.27** food items. The mean paired
difference was **−9.09**. B did **not** outperform A: its primary outcome was
lower for all three seeds under this protocol. This does not establish that
distance features generally harm learning or could never help another setup.

Secondary metrics, including medians, within-policy score SD, mean lengths,
and collision/truncation/win rates, are in the
[complete report](results/milestone5d_feature_comparison/report.md) and raw CSVs.
For example, B/seed 22 survived longer on average (691.12 versus 552.41 moves)
while scoring less; it truncated in 51% of games. Longer episodes alone do not
establish better food collection. No evaluated model won a game.

The four saved charts are:

- [Average evaluation score](results/milestone5d_feature_comparison/average_score.png).
- [Rolling training scores](results/milestone5d_feature_comparison/training_performance.png).
- [Evaluation episode length](results/milestone5d_feature_comparison/episode_length.png).
- [Between-training-seed variability](results/milestone5d_feature_comparison/performance_variability.png).

Validation included **21 focused comparison tests**, relevant existing checks,
and a separate two-mode development smoke test (225 finite updates per mode).
The full suite ran once: **229 passed, 0 failed**, with one dependency
deprecation warning. Saved run hashes, all 600 evaluation records, paired
configurations, and aggregate calculations were verified independently.
All 31 tracked historical artifacts match Git, and the original seed-7 final
checkpoint SHA-256 remains unchanged. The original model was not retrained, no
historical results were regenerated, and no dependencies were added. Milestone
5D remains uncommitted and has not been pushed.
