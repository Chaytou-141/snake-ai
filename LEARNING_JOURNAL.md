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
