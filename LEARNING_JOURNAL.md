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

## Milestone 3: understanding the neural network

We now have `DQNNetwork` in `snake_ai/models/dqn.py`. Its name describes its
future role: predicting action values for DQN. It is only a neural-network
building block today. No Snake training algorithm has been implemented.

### Neurons: a weighted calculation

A neuron combines input numbers with weights and a bias. For inputs x1 and x2,
its basic calculation is `z = w1*x1 + w2*x2 + b`. With inputs 1 and 0, weights
0.4 and -0.2, and bias 0.1, the result is 0.5. With 11 inputs it sums 11 weighted
terms instead. This is an arithmetic unit, not a simulation of a biological brain.

In SnakeAI, the first layer receives the 11 danger, heading, and food flags. One
neuron has a separate weight for each flag and its own bias. The bias shifts the
result even when every input is zero. `nn.Linear(11, 128)` computes 128 such
weighted sums at once. PyTorch stores the weights as a matrix with 128 rows and
11 columns, and the biases as a vector of length 128.

### Weights: adjustable numerical parameters

Weights and biases are model parameters. They begin with random initial values;
`torch.manual_seed(7)` makes initialisation repeatable on our installation.
Learning adjusts the parameters using data and a loss. A weight's magnitude
affects how strongly an input contributes, but one weight alone generally does
not explain the whole decision because later layers combine many activations.

`DQNNetwork.parameter_counts()` uses `parameter.numel()` to count stored numbers.
The first layer has `11*128 + 128 = 1536`. The second has
`128*128 + 128 = 16512`. The last has `128*3 + 3 = 387`. The measured total is
18,435 trainable parameters. ReLU has no trainable weights of its own.

### Layers: moving from observations to action values

The input is the 11-value observation tensor; it does not add trainable
parameters by itself. The two hidden layers each contain 128 neurons. Hidden
means these are intermediate calculations, rather than the final answer.
The output layer has three neurons, corresponding to straight, right, and left.

The architecture is `11 -> 128 -> 128 -> 3`, with ReLU after each hidden layer.
We can change `input_dim`, `hidden_size`, or `output_dim`, but the defaults match
the existing environment. A single input has shape `(11,)`; a batch of four
inputs has shape `(4, 11)` and produces output shape `(4, 3)`. Shape describes
how many entries exist along each axis, not what their values mean.

### ReLU: introducing a nonlinear activation

ReLU calculates `max(0, z)`: negative values become zero and positive values
remain unchanged. For example, -0.3 becomes 0 and 0.5 stays 0.5. An activation
function lets the network model nonlinear relationships. Without nonlinear
activations, composing several Linear layers would still be equivalent to one
affine transformation, even though it contained more arithmetic.

Our `forward()` uses `self.relu()` after the first two Linear layers. There is
no ReLU or Softmax after the final layer because Q-values must be able to have
either sign and arbitrary magnitude.

### Forward propagation: calculating predictions

`model(inputs)` calls the module's `forward()` method. The first layer combines
the observation flags, ReLU changes its results, the second layer combines
those activations, and the last layer returns three numbers. This calculation
is forward propagation. It does not change the weights or teach the network.

`snake_ai/demos/network.py` resets the real environment with seed 42 and uses
`torch.tensor(list(observation), dtype=torch.float32, device='cpu')` to convert
the standard-library array into a tensor. A tensor is PyTorch's numerical array;
float32 uses a 32-bit floating-point value for each number. The demo calls
`model.eval()` and `torch.no_grad()` for inference. Eval mode controls the
behavior of modules such as dropout if we add them later; no_grad disables
gradient tracking. Neither operation trains the model.

For this fixed example the untrained outputs are approximately
`[-0.00737532, -0.12700197, -0.03377551]`. `argmax` returns the index of the
largest number: index 0, straight. All three can be negative and one still has
the highest value. The resulting environment move earns reward 0 and continues
the episode. This is a mechanical demonstration, not a learned decision.

### Loss: measuring prediction error

In `snake_ai/demos/optimisation.py`, we make a batch of three fixed feature rows
and invent three target triples. For one row a target is `[1, 0, -1]`. These
are just teaching labels, not Snake rewards or Bellman targets. The example is
ordinary supervised regression: predict specified numerical values.

`nn.MSELoss()` calculates mean squared error. For each output it subtracts the
target, squares the difference, then averages over all nine outputs in the batch.
If a prediction is 0.2 and its target is 1, that entry contributes
`(0.2 - 1)^2 = 0.64` before averaging. Squaring makes either sign of error
positive and gives larger errors more influence. Loss is a scalar measuring
error against our chosen labels, not the game's displayed score.

### Gradients: local sensitivity of the loss

A gradient describes how a small parameter change affects the loss locally.
For a particular weight, a positive derivative means increasing it slightly
tends to increase the loss nearby; moving it in the opposite direction tends
to help locally. Gradients are not guarantees about large changes.

For the simple squared error `(prediction - target)^2`, the derivative with
respect to the prediction is `2*(prediction - target)`. PyTorch also accounts
for averaging and all the calculations between a weight and the prediction.
Each parameter tensor has a `.grad` tensor containing those derivatives after
backpropagation. Our example reports an aggregate gradient L2 norm of about
0.64655010, calculated by squaring and summing all gradient entries and taking
the square root. A nonzero norm shows that there is a direction for an update;
it is not a Snake performance metric.

### Backpropagation: calculating the gradients

While the forward pass runs with gradient tracking enabled, PyTorch records how
the calculations depend on trainable parameters. `loss.backward()` works back
through this computation graph using the chain rule, which combines derivatives
through a sequence of operations. We do not manually derive each layer's weights.

Backpropagation calculates gradients; it does not itself update the weights.
PyTorch accumulates gradients by default. `optimiser.zero_grad()` clears old
ones before `backward()` so we use this batch's derivatives. The model tests
verify that gradients exist, are finite, and include nonzero entries.

### Optimiser: applying a parameter update

The demo creates `torch.optim.SGD(model.parameters(), lr=0.001)`. In this simple
configuration, stochastic gradient descent updates each parameter by
`new_weight = old_weight - learning_rate * gradient`. Biases are adjusted too.
For a weight 0.4 and gradient 0.2, this gives `0.4 - 0.001*0.2 = 0.3998`.
The learning rate controls the size of the step.

The actual code performs a forward pass, calculates MSE, clears gradients,
calls `loss.backward()`, and calls `optimiser.step()` exactly once. It then does
a new forward pass on the same batch to measure loss after the update. Measuring
the old `loss` tensor again would still report the old calculation.

With seed 7 and learning rate 0.001, loss changed from 0.5388405919 to
0.5384227037. All six trainable parameter tensors changed: weights and biases
for each of three Linear layers. One optimiser step does not guarantee a lower
loss for every configuration. Even this measured decrease only fits our invented
labels a little better; it teaches us nothing about the model's ability to play.

### Q-values: the eventual meaning of the outputs

After future reinforcement-learning training, a Q-value is intended to estimate
expected discounted cumulative reward for taking one action from an observation
and then following a policy. Discounting gives rewards further in the future
less weight according to a chosen discount factor. It is not a probability of
choosing an action, nor just the immediate reward for the next move.

Our three output positions correspond to `0=straight`, `1=right`, and `2=left`.
Standard DQN treats these as regression values. Softmax would force positive
values summing to one, removing the freedom to express the value scale and
negative returns. That is why `output_layer` returns raw numbers.
For now, these outputs come from random initialisation; calling them Q-values
describes their intended role, not evidence that they estimate rewards accurately.

### Scope and reproducibility

The network demo takes one real environment step. The optimisation demo never
uses environment transitions and never builds DQN targets. We have no replay
memory, epsilon-greedy exploration, target network, Bellman update, or training
loop. The two demos create separate fresh models and save no model files.

CPU-only PyTorch 2.15.0+cpu was installed in the existing Python 3.12 environment.
Its optional NumPy bridge warns because NumPy is absent; our list-to-tensor
conversion does not need that bridge. A Pygame packaging deprecation also appears
in the full test suite. These warnings do not prevent the demos or tests running.
Same seeds reproduce model initialisation and outputs on this installation;
different library versions, devices, or platforms can still produce differences.

The 22 new model/demo tests and all 88 earlier tests pass: 110 in total. The
existing engine, renderer, environment, and baseline files were preserved.
The limitations of partial observations still apply; a neural network cannot
reconstruct missing board information simply by having more neurons. Milestone
4 will need approved learning-algorithm work and decisions about truncations,
rewards, and observations before any claim of trained Snake performance.
