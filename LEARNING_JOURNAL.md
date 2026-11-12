# Learning journal — Milestone 1

## Classes and objects

A class describes a kind of object. `SnakeGame` in `snake_ai/game/engine.py`
describes a board and its rules. `game = SnakeGame(seed=42)` creates an object:
one particular game. Creating another object gives it its own body, score, and
random-number generator. `Direction` and `GameStatus` are enums: named choices
that prevent confusing strings such as `"up"` and `"UP"`.

## Variables and game state

Variables give names to values. An object's attributes keep values between
method calls. `game.snake`, `game.food`, `game.direction`, `game.score`, and
`game.status` together describe the current game state. Inside `step()`,
`next_head` is a temporary local variable used to calculate a possible move.
`reset()` restores the starting state. A terminal status means the game has won
or lost; more steps do nothing until a reset.

## Functions and methods

A function packages a task into reusable code. A method is a function belonging
to a class. `SnakeGame.step()` advances one move; `SnakeGame._spawn_food()`
selects food; `Direction.is_opposite()` checks whether a turn reverses direction.
`self` refers to the particular object receiving the call. The underscore in
`_spawn_food` marks it as an internal helper. `draw()` in `renderer.py` is a
regular function because it does not belong to a class.

## Lists and tuples

The snake is a list of positions. Lists can change: `insert(0, next_head)` adds
the head and `pop()` removes the tail. When food is eaten, skipping `pop()` makes
the snake grow. Each position is a tuple, such as `(10, 10)`: a fixed pair of
values. A list is useful for a body whose length changes; tuples are useful for
individual coordinates. `snake[:-1]` means every segment except the tail.

## Grid coordinates

The game thinks in cells, not pixels. `(0, 0)` is the top-left cell, and `(19, 19)`
is the bottom-right cell of the default board. `Direction.UP` contains `(0, -1)`.
Adding those values to `(10, 10)` gives `(10, 9)`. Only the renderer converts cells
to pixels using `CELL_SIZE` and adds `HEADER_HEIGHT` for the score area.

## Collision detection

`step()` calculates the next head before changing the body. It checks whether
the new coordinates are outside the board or inside the occupied body. The tail
is excluded from the body collision check on a normal move because it leaves
its cell simultaneously. When eating, the tail stays, so the entire body counts.
On collision, the game records `LOST` while retaining its last valid body. After
the last empty cell is eaten, the game records `WON` and stops spawning food.

## Random seeds

Random generators produce a repeatable sequence from a starting seed. Each
`SnakeGame` owns a `random.Random` object rather than changing global randomness.
`_spawn_food()` lists empty cells in a stable order and chooses one randomly.
Two games with the same seed and calls produce the same choices. `reset()` keeps
using the next random numbers; `reset(seed=42)` starts the sequence again. This
will let us compare future agents under matching starting conditions.

## Separation of responsibilities

`engine.py` owns rules. `renderer.py` owns input, drawing, and timing.
`__main__.py` reads command-line options and starts the game. The renderer sends
directions to `step()` and reads the resulting state. The engine never imports
Pygame. This means a human, a test, or eventually an agent can all use the same
rules. No neural network or learning algorithm is required to achieve this.

## Automated testing

A test is code that checks expected behavior. Pytest finds functions named
`test_*` in `tests/test_engine.py`. An `assert` compares actual and expected
results. `test_moving_into_vacated_tail()` builds a precise body shape and checks
a subtle legal move. `test_full_grid_wins_without_spawning_loop()` uses a tiny
2 by 2 board so the winning case is easy to understand. Parameterized tests
repeat a check for several directions or invalid dimensions.

Some tests intentionally assign state to arrange a particular scenario; normal
gameplay should use the public `step()` and `reset()` methods. Fixed seeds and
explicit fixtures avoid flaky tests. A subprocess test confirms that importing
and stepping the engine does not even load Pygame. Rule tests run without a
window; they cannot prove that a real display and keyboard feel correct.

## Next learning step

Milestone 1 stops at a playable game. Later, an adapter can turn game state into
observations and agent decisions into directions. Rewards and neural networks
will be separate additions, leaving the tested movement rules in place.

## Milestone 2: reinforcement learning

Reinforcement learning means learning to choose actions through interaction and
feedback. An agent tries an action, the environment advances, and a reward tells
the agent something about the result. Milestone 2 builds this interaction but
does not learn: `RandomAgent` always samples actions randomly, without updating
any knowledge. PyTorch and DQN still belong to a future milestone.

### Agent and environment

`RandomAgent.act()` chooses a number using its own seeded random generator.
`SnakeEnv.step()` receives the number, converts it to a direction, and asks the
existing `SnakeGame` to move. The environment also tracks episode limits and
rewards. The agent makes decisions; the environment applies rules. Neither
needs to open a Pygame window.

### State and observation

State is everything that describes the game: body coordinates, food, direction,
score, and status, plus the wrapper's episode counters. An observation is the
information we give the agent. `SnakeEnv.observation()` turns part of the state
into 11 binary float32 numbers in a standard-library `array('f')`.

The order is danger straight/right/left, moving up/down/left/right, and food
left/right/above/below. A 1 means true and a 0 means false. A snake moving right
with food above and to the right on an otherwise safe next move sees
`[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]`. `FEATURE_NAMES` records this exact order.

`SnakeGame.would_collide()` predicts immediate danger using the same rules as
`SnakeGame.step()`. A departing tail is safe to enter. Sharing this method
prevents the observation and the actual game from disagreeing about that rule.

### Action space

An action space is the set of allowed decisions. Our action space is three
integers: 0 continues straight, 1 turns right, and 2 turns left. These actions
are relative to the head, rather than fixed screen directions. For example,
`relative_direction(Direction.RIGHT, 1)` is `Direction.DOWN`; when facing up,
the same right-turn action means `Direction.RIGHT`. Reversing is impossible.
Invalid actions raise an error instead of silently choosing a different move.

### Reward functions

A reward is a number describing one step's outcome. `RewardConfig` starts with
+10 for food, -10 for collision, 0 for normal movement, and +10 for winning.
The final meal pays only the win reward, not both rewards. These numbers are
design choices that may need experiments later, not facts about the best way
to train. A high reward does not change the displayed food score: the score
still counts meals, one point each.

### Episodes, termination, and truncation

An episode starts with `env.reset()` and ends at a game outcome or time limit.
Termination means a natural outcome: collision or a filled board. Truncation
means we stopped a still-running game because a chosen budget was exhausted.
`SnakeEnv.step()` returns these as separate booleans.

The default total budget is 10,000 moves. Starvation allows at most
`max(400, 100 * len(env.game.snake))` consecutive moves without food. Initially,
length 3 gives 400. Eating sets `steps_since_food` to zero. With length 5, the
allowance is 500. An endless circle therefore ends even if it avoids collisions.
Truncation does not invent a collision penalty. A collision on the very last
allowed move remains a natural termination, and a win remains a win.

The environment reports `end_reason` and counters in the `info` dictionary.
For a truncated episode the underlying game can still be `running`, but the
wrapper requires a reset. A future learner must decide how to handle value
estimates at a time limit; treating every truncation as death can distort them.

### Random policies and baselines

A policy is a rule for choosing actions. `RandomAgent` is a uniform random
policy: each action has probability one third, even if it is dangerous.
It ignores observations and never improves. Two agents with the same seed
produce the same action sequence; the environment's separate seed determines
the food sequence. `evaluate()` uses documented seed pairs for each episode.

A baseline gives future learning results a comparison. The measured 100-episode
run averaged 0.14 meals and 66.49 moves; every episode ended in collision.
This is an initial sample, not a precise estimate for every possible random
run. `results/random_baseline.csv` records each episode, and the adjacent JSON
stores the seeds, settings, and summary so a claim can be checked. Repeating
the experiment matched all rows. The evaluator refuses to overwrite experiments.

### Limits of these 11 features

The features do not show the entire snake, distance to food or walls, free
space, or remaining episode budgets. Two different boards can produce identical
observations even though the best action differs. This is partial observation:
the input is not a complete Markov state, which would contain enough information
to predict the next-state distribution from an action. Immediate safety also
does not guarantee that a move avoids a trap several steps later. A future DQN
cannot recover information we never give it simply by being a bigger network.

### What the tests teach us

`test_environment.py` checks all direction/action conversions, exact observation
order, and predicted danger against actual engine moves. Small, controlled
boards test rewards, wins, starvation, and step caps. `test_evaluation.py` checks
seeded action sequences, metrics, CSV/JSON output, and overwrite protection.
A subprocess runs evaluation and verifies that it imports neither Pygame nor
ML libraries. All 33 Milestone 1 tests still pass, helping preserve human play.

Milestone 2 stops here. Before training, we still need choices about observations,
rewards, tensor inputs, and time-limit handling, followed by the actual network
and learning machinery in later milestones.
