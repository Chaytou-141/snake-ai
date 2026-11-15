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

## Milestone 4: learning action values from gameplay

### What DQN adds

A Deep Q-Network uses a neural network to estimate the value of actions, then
adjusts those estimates using actual transitions collected from the environment.
In Milestone 3 we practised changing weights toward invented labels. In Milestone
4, `DQNAgent` in `snake_ai/agents/dqn_agent.py` builds learning targets from
gameplay: current observation, action, reward, next observation, and end flags.

The neural network is still 11 -> 128 -> 128 -> 3 with ReLU hidden activations
and raw outputs. The new parts are replay memory, an exploration policy, a
second network for targets, and a training loop. This is standard DQN; it does
not include Double DQN or later milestone improvements.

### An experience and replay memory

Suppose the snake is moving right and we choose action 2, a left turn toward
the top of the screen. `SnakeEnv.step(2)` supplies the resulting observation,
reward, and end flags. Together with the original observation and action,
that forms one experience. `DQNAgent.remember()` copies it into `ReplayBuffer`
and advances the environment-step counter.

`ReplayBuffer` uses `deque(maxlen=50000)`, so the oldest item leaves when memory
is full. Observations are copied into tuples, avoiding a bug where changing the
current observation would accidentally change an older experience. A sample
becomes float32 state/reward tensors, int64 action indices, and boolean flags.

Consecutive Snake moves are related: one follows directly from another in the
same part of the board. Learning only from that sequence can overemphasise a
recent situation. `sample(64)` mixes experiences from different times, reducing
these correlations and allowing reuse of previous gameplay. Sampling is without
replacement within a batch, but an experience can appear in many different
batches. A separate seeded RNG makes sampling reproducible without interfering
with food randomness or exploration.

### Exploration: gathering information through random actions

If we always followed an untrained network's highest output, we might repeat
the same poor choices and see little of the game. Epsilon-greedy adds exploration:
with probability epsilon choose a uniformly random action; otherwise use argmax.
It does not avoid dangerous actions during exploration. Random collisions are
part of the information used to learn.

Epsilon starts at 1.0. It decreases linearly to 0.05 over 50,000 collected
environment moves, remaining at 0.05 afterward. At 25,000 moves it is 0.525.
Environment moves and optimiser updates are different counters: the first 999
moves collect data without updates, then learning starts at move 1,000 when
enough replay samples exist. Predicting an action or doing an extra training
update does not itself advance epsilon.

`act(observation, evaluate=True)` ignores training epsilon and acts greedily.
It neither changes the schedule nor consumes the exploration RNG. Evaluation
therefore measures the learned policy without random exploratory failures and
does not change what the next training action would have been.

### Updating the value of the action actually taken

The online network returns three numbers, but each experience tells us what
happened after only one selected action. `selected_q_values()` uses PyTorch's
`gather` to select that output. For predictions `[2, 4, 1]` and action 0, the
prediction being trained is 2, not the maximum 4 and not all three values.
Only the selected output receives a direct error signal from that transition;
shared hidden weights can still affect the other predictions after an update.

The standard DQN target is:

```text
target = reward + gamma * (1 - terminated) * max Q_target(next_observation)
```

For example, if food gives reward 10, gamma is 0.99, and the target network's
best next-state value is 4, the label is `10 + 0.99*4 = 13.96`. This accounts for
both the food just eaten and the estimated reward still available afterward.
For a collision giving -10, genuine termination makes the label exactly -10,
regardless of any network output. There is no future gameplay after death.

For a normal move truncated only by a time limit, reward 0 and next value 4
give `0 + 0.99*4 = 3.96`. We preserve the bootstrap because we chose to stop the
episode for a budget reason; the game did not naturally end. The environment's
starvation limit is also treated as an external truncation. This is a modelling
choice and is documented, especially because the observation hides that timer.
The next observation comes from before reset, avoiding accidental learning
across unrelated episodes.

### Gamma: valuing the future

Gamma, or the discount factor, determines the weight given to later rewards.
With gamma 0.99, a reward one step later is weighted by 0.99, two steps later
by `0.99*0.99`, and so on. A smaller gamma focuses more on immediate reward; a
larger one places more weight on future possibilities. Our gamma 0.99 is an
experimental starting point, not an established best value for Snake.

### Online and target networks

The online network makes action predictions and receives gradient updates. If
we also changed all the quantities used to create its labels at every update,
the learning problem could become unstable: a moving model would chase rapidly
moving labels based on its own predictions.

The target network is a separate copy whose weights stay fixed for a while.
`compute_targets()` uses that copy to estimate future value. Every 1,000 optimiser
updates, `load_state_dict()` copies the online weights into it. This provides a
temporarily steadier reference; it does not make the estimates correct or
guarantee stable learning. Our target is frozen with `requires_grad_(False)` and
target calculations also run under `torch.no_grad()`. It receives no gradients
and is not part of the Adam optimiser.

This implementation uses the target network both to identify and value the
largest next action. Separating those two roles would be Double DQN, which is
outside this milestone.

### Loss, gradients, and the optimiser

The selected online predictions and detached targets are compared with
`smooth_l1_loss`. Smooth L1, also called Huber loss with the default transition
scale, is quadratic near zero error and linear for larger errors. It reduces
the influence of large errors compared with plain squared error.

The training step clears gradients, calls `loss.backward()`, clips the combined
gradient norm to at most 10, and calls Adam's `step()`. Backpropagation tells us
how changing each parameter affects the current batch loss. Adam uses gradients
and its internal running estimates to update parameters at learning rate 0.0005.
Clipping controls unusually large updates; it does not guarantee that the model
will be stable or that every update will help gameplay. The CSV records the
norm before clipping so clipping activity can be inspected.

Loss measures disagreement with our current bootstrap labels, which themselves
change as the target network changes. It is not a direct score or a fixed measure
of correctness. Better game performance can coexist with noisy or increasing
loss as the replay distribution and target values change. We must evaluate the
policy, rather than declare learning merely because a loss curve falls.

### Training stages and actual evidence

We first ran all existing tests before committing Milestone 3 as `9a8e1e0`.
Then unit tests checked DQN targets, replay, exploration, gradients, target copies,
and checkpoints. A separate smoke run collected 500 experiences, made 437 finite
updates, changed weights, and loaded its checkpoint successfully. A greedy
evaluation check left weights and training progress unchanged.

The initial experiment then used the fixed configuration in `configs/dqn.json`:
seed 7, 500 episodes, and a 100,000-move cap. It completed all 500 episodes in
54.23 seconds, using 33,629 moves and 32,630 updates. Training food seeds were
100000 through 100499. Mean training score was 1.05, median 1, minimum 0, maximum
8; the last 100 episodes averaged 1.96. Epsilon was still 0.361049 at the end,
so random actions continued to shorten many training episodes. We did not
extend the budget or search for better hyperparameters.

The final checkpoint was evaluated greedily on food seeds 900000 through
900099, reserved separately from training. A fresh random policy used those
same environment seeds and its own action seeds 2000000 through 2000099. The
two policies visit different states and need not consume food randomness at the
same times; matching seeds provides paired starting conditions, not identical
full trajectories.

The DQN's mean score was 34.81, median 37, minimum 8, and maximum 58. Random's
mean was 0.14, median 0, minimum 0, and maximum 1. Average episode lengths were
637.28 and 59.64 moves. Every episode for both policies ended in collision:
100% collision, 0% truncation, 0% win. The mean score difference was +34.67.
That is measurable improvement in this comparison, but not a solved game or
proof that other training seeds would produce the same result.

The old random baseline also averaged 0.14 by coincidence, but it used different
seeds and had a maximum of 2. It remains unchanged; the new paired results live
in `results/milestone4_evaluation/`. This separation prevents confusing an old
experiment with a fresh comparison. The checkpoint hash, seeds, and settings
are included in the new evaluation summary.

### Saving progress and reading charts

`run_training()` saves episode and update CSVs, configuration, summary, static
Matplotlib charts, periodic checkpoints, and a final checkpoint. Each experiment
uses a new directory. Checkpoint paths refuse overwriting, and `.pt` files are
ignored by Git. The five plots use episode axes for score/reward, optimiser
steps for loss, and environment steps for epsilon. The rolling score plot uses
up to 25 completed episodes; a partial final episode is clearly labelled in
the CSV and omitted from completed-episode statistics.

Ctrl+C preserves a final checkpoint and collected metrics. The checkpoint stores
both networks, Adam state, counters, configuration, seeds, and some RNG states.
Default loading is for frozen inference. `resume_training=True` is an explicit
request for approximate continuation: replay memory and the active episode are
not restored, so the replay sampling history cannot be reproduced. Counters and
epsilon continue, a new episode uses the next training seed, and replay refills
from scratch. We cannot promise an exact continuation of an interrupted run.

Final verification: 36 focused tests and 146 total tests passed. Matplotlib
installed NumPy as a dependency, and the earlier optional torch/NumPy integration
warning is resolved; the existing Pygame packaging warning remains. No game rules,
rewards, or observations changed. The 11-feature partial observation, modest
single-seed experiment, and approximation on resume remain limitations. We stop
at Milestone 4; further training strategies require a later approved milestone.

## Milestone 5A: watching the saved model make decisions

### Training versus inference

Training uses experiences, loss, backpropagation, and an optimiser to change
weights. Inference uses the existing weights to calculate predictions. Watching
the trained AI is inference only. `snake_ai/demos/trained_ai.py` never calls
`remember()`, `train_step()`, or the optimiser. It saves no model files and does
not change the earlier experiment's CSVs or summaries.

`PlaybackSession` sets the model to evaluation mode with `model.eval()`, freezes
its parameters, and uses `torch.inference_mode()` during forward passes. Eval
mode controls training-specific behavior if layers such as dropout are present;
our current Linear/ReLU network has no such layers. Inference mode disables
gradient tracking and additional autograd bookkeeping. Neither command improves
the learned policy: they tell PyTorch how to run predictions efficiently.

### Loading a saved network

The checkpoint at `results/milestone4_initial/checkpoints/final.pt` stores the
trained weights along with configuration and training progress. The demo calls
the existing `DQNAgent.load()` method, which reconstructs `DQNNetwork` and restores
its tensors with `load_state_dict()`. Loading uses CPU mapping and the supported
checkpoint format. Shapes must match: our verified saved model has 11 inputs,
two hidden layers of 128 neurons, and three outputs, containing 18,435 numbers.

The demo also creates `SnakeEnv` with the environment/reward settings recorded
in the checkpoint. This matters because silently changing rewards, observation
meaning, or action conventions could make the model's estimates misleading.
The same 11-feature encoder and relative action conversion remain in use. The
model file is not copied over or saved again during playback.

### From observations to Q-values

`env.reset(seed=800000)` supplies a real initial observation. On this installation,
it is `[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]`: immediate moves are safe, the snake
faces right, and food is rightward and below. `_refresh_decision()` converts it
to a float32 CPU tensor and sends it through the saved network.

The actual initial outputs are approximately `[3.5001304, 3.1838040, 3.2873244]`.
Their positions mean Straight, Right, and Left, in that order. `argmax` chooses
index 0 because 3.5001304 is the largest value. `PlaybackSession.step()` passes
that integer directly to `env.step(0)`, so the environment applies its usual
relative action rule: continue RIGHT for this example. After receiving the new
observation, the session calculates the next decision.

The running panel shows values for the currently displayed board, not stale
values from the board before the last move. The highlighted action will execute
on the next movement tick. Pausing keeps that board and decision visible. When
an episode ends, the last executed decision is retained and labelled **FINAL
MOVE Q-VALUES** and **Values before final move**. There is no valid next action
in a finished episode, so we do not pretend those values describe another move.

### Why Q-values are not probabilities

A Q-value estimates cumulative discounted reward for taking an action and then
following the policy. A probability lies between zero and one; action
probabilities in a distribution sum to one. Our initial outputs are each above
three and sum to more than nine. That is allowed for action values and clearly
does not represent a probability distribution. Other observations can produce
negative estimates, such as the value of moving into a wall.

The demo displays the raw values, highlights the largest, and does not use
Softmax or percentage bars. A larger estimate means the model currently expects
more return from that action, not that it is guaranteed safe or has a particular
probability of success. Greedy means choosing the largest estimate without
random exploration; it does not mean the action is objectively optimal.

### Controls and repeatable episodes

Pygame receives keys and controls the timing; the model makes all movement
decisions. Space toggles pause. Plus and minus change moves per second within
1–60. Neither control changes an observation, reward, or model weight. Arrow
keys do not steer the AI, keeping the displayed behavior attributable to the
network rather than mixing in human decisions.

R calls `reset()` using the current seed, keeping the episode number. Repeating
the same initial conditions with this greedy policy reproduces the trajectory
on the same installation. N starts the next episode with the next seed and
increases the episode number. Both clear score, reward totals, move counts, and
terminal flags. The interface waits after either termination or truncation so
we can inspect the final result before choosing R or N.

### Why a trained agent still makes mistakes

The network estimates values from a compressed observation, not the complete
board. Different body shapes can produce the same 11 features but require
different long-term plans. Avoiding an immediate collision does not reveal a
trap several moves away. The finite training sample and approximate bootstrap
targets can also produce inaccurate estimates. Taking argmax cannot correct
the information or estimation errors underneath those numbers.

The model's original held-out scores improved over random, but every evaluated
episode still ended in collision. Playback makes those successes and failures
visible; it does not solve them. An 80-move headless playback check on seed
800000 collected seven food items and rendered a readable preview, with model
weights and checkpoint hash unchanged. This checks the interface, not a new
claim about policy performance.

### What was preserved and tested

Before adding playback we verified all 146 Milestone 4 tests, the checkpoint's
architecture/hash, and the saved training/evaluation metrics. Milestone 4 was
committed as `5067a30`. The new 27 tests exercise temporary checkpoints and a
headless Pygame display, including actual inference mode, no weight updates,
all heading/action mappings, pause, retry, new episodes, speed controls, and
natural versus truncated endings. All 173 project tests pass. Only the existing
Pygame packaging deprecation remains.

The original renderer accepts optional display labels and overlay control for
reuse in the AI view; its default human-play rendering and controls remain the
same. No training algorithm, model architecture, reward, saved checkpoint, or
evaluation result changed. Milestone 5A stops at visual inference; further
changes belong to a separately approved milestone.

## Milestone 5B: learning from collision evidence

### Why a trained model can still crash

Training improved our policy's scores, but it did not prove that the agent can
avoid every trap. The model only receives 11 features, not the full board or
snake-body geometry. Its Q-values are approximate estimates learned from a
finite set of experiences. Either limited information or imperfect estimates
could matter. Diagnostics help us examine recorded events instead of assuming
which explanation caused a particular failure.

The new module is `snake_ai/evaluation/collision_diagnostics.py`. It uses the
existing checkpoint loader, `agent.predict()`, `env.observation()`, relative
action conversion, `would_collide()`, and `env.step()`. It gives the model no
extra information. Complete bodies are recorded for our analysis only.

### What a diagnostic records

`record_move()` copies the state immediately before the action: head and body,
heading, food, score, length, and observation. It records the three Q-values and
their argmax, checks each relative action against the existing collision
predictor, then takes the selected action through the environment. Finally it
copies the actual resulting state, reward, and end flags/reason.

Copying matters. If a log simply kept a reference to a changing body list, later
moves could alter our evidence about an earlier moment. A `deque(maxlen=30)` in
`diagnose_episode()` keeps only the latest 30 copied moves. We therefore retain
useful local context without storing the entire trajectory of every episode.

On a collision, our engine leaves the head at its last valid cell. For example,
a head at [19, 0] can attempt to move outside the wall while its actual post-action
position stays [19, 0]. `attempted_head` is explicitly a projected destination;
`head_after` and `body_after` come from the real engine state, not a reconstruction.

### Immediate safety versus long-term planning

The predictor answers a narrow question: would this one move hit a wall or body,
allowing the tail's departing cell when appropriate? It does not simulate all
possible futures. A move can be safe now but contribute to a trap later. Seeing
a collision later does not automatically make every preceding move a mistake.

`classify_end()` gives the final event one of three labels. Type A means the
agent chose an immediate collision while another action was immediately safe.
Type B means no safe next action existed. Type C covers other endings or evidence
that cannot be trusted consistently. A Type B collision may have been preventable
earlier; the label says only that the final state offered no safe next move.

### Why argmax can select a dangerous move

Argmax returns the largest estimate even when every available action is bad.
It does not create a safe action. In episode 1, seed 900000, the final Q-values
were approximately Straight -10.175884, Right -9.489347, Left -10.461297.
Right was highest, so the greedy agent chose it. The collision predictor marked
all three actions dangerous. There was no immediately safe final alternative.

A Q-value estimates future discounted reward, not a probability or an immediate
safety signal. Negative values can occur for safe actions too. If a safe
alternative exists, we can compare its Q-value with the selected value. If none
exists, that comparison is undefined; our JSON uses `null` and CSV uses blank
scalar cells, rather than inventing a gap of zero.

### What actually happened in the 100 episodes

Before drawing conclusions, `verify_reproduction()` checked every episode's
score, move count, reward, end flags, and end reason against the original saved
evaluation. It also compared the complete aggregate metrics. All reproduced:
mean score 34.81, median 37, maximum 58, and mean episode length 637.28 moves.
Each episode ended in collision. All 100 were Type B; none were Type A or C.

This means our sample does not show final-step failures where the agent ignored
an immediately safe alternative. It does not prove the network planned well
earlier, or that the observation representation caused the trap. The saved final
30 moves allow further inspection, but the setup may have begun before them.

The requested Type A example could not be supplied because no such event was
observed. The detailed examples instead include the first Type B event (episode
1, score 40, collision on move 838) and the maximum-score event (episode 35,
seed 900034, score 58, collision on move 1372). In the latter, Right had the
highest Q-value, approximately -9.271073, but all three choices collided. Missing
examples are explicitly absent rather than manufactured from another dataset.

Selected final Q-values averaged -9.417852, with a range from -10.043816 to
-8.523903. They were all negative in this particular sample. We used the engine
to establish danger; we did not infer danger from the sign of the numbers.

### How failure analysis guides a future experiment

The results make earlier trap formation a useful next question. A future,
separately approved analysis could define measurements of reachable space or
longer-term options and identify when those options shrink. It could then
motivate a controlled comparison of observation or learning choices. That would
be a hypothesis to test, not an improvement established by today's logs.

The current seeds have already been evaluated and inspected, so future model
comparisons should use fresh untouched evaluation seeds and independent training
seeds. We should define the comparison before looking at those new results.
This milestone performs no retraining, reward/feature changes, algorithm changes,
or multi-seed training experiments.

### Files and checks

`results/milestone5b_collision_diagnostics/` contains the summary, episode rows,
3,000 retained moves, final collisions, detailed JSON examples, a Markdown report
with the schema, and a static chart. The runner refuses existing output directories
and stops before publishing analysis if reproduction fails. Model parameters,
training counters/RNG, checkpoint bytes, and reference files are checked for
preservation. All 23 older saved files were preserved, and all earlier gameplay,
training, evaluation, and playback source files remain unchanged.

The 17 focused diagnostics tests and full 190-test suite pass. The only warning
is the pre-existing Pygame packaging deprecation. Earlier journal commit hashes
refer to the historical local backup; the README now lists the equivalent hashes
published after the authorised email rewrite. Milestone 5B stays uncommitted and
is not pushed until reviewed.

## Milestone 5C: following the formation of a trap

### Why analyse earlier states?

Milestone 5B found no safe next move at any of the 100 final collisions. That
does not tell us when a trap formed or whether an earlier alternative would
have avoided it. We therefore kept the same frozen model and examined every
pre-move state in the same episodes, using `trap_analysis.py`. Before this work,
we verified the 190-test suite and checkpointed 5B as `48709f81cbb2a970a7f4d389354d5b90d3514fcb`.
The checkpoint was local at the start of Milestone 5C.

### Immediate danger is a one-step question

For each relative action, the existing engine predicts whether the next move
would collide. `measure_state()` counts how many predictions are safe. Three
means all available actions are legal now; zero means none is legal. It does
not mean that three safe choices guarantee a successful long-term plan.

We record the first movement with exactly two, one, and zero safe choices. We
also count transitions that reduce choices and transitions back from one choice
to two or three. A narrow corridor can lead to an open region, so a reduction
is a measurement, not a diagnosis of a mistake.

There were 1,704 consecutive runs of one-safe states; 1,628 later returned to
at least two options, while 76 reached zero. Thus 95.54% of those runs recovered.
At a different level, 1,628 of the 3,126 individual one-safe visits recovered
on the very next move, or 52.08%. Naming the denominator prevents a misleading
percentage. These are repeated states within episodes, not independent trials.

### Measuring distance to an obstacle

`free_cell_distance()` traces a ray until it encounters a wall or the current
body. It counts free cells, excluding the starting head. On a 5 by 5 board,
a lone head at (2,2) has two free cells to the right before the wall: (3,2)
and (4,2). The next grid step is outside the board. An obstacle immediately
adjacent to the head gives a distance of zero.

This geometric ray treats the current body, including the tail, as static.
The collision predictor still correctly allows a move into a departing tail.
These answers can differ without a bug: one describes current geometry, while
the other applies the moving-snake rule. Across the measured states, mean ray
distances were 7.960 straight, 7.468 right, and 6.993 left free cells.

### Breadth-First Search: exploring connected cells

A grid can be viewed as a graph. Each free cell is a node, and an edge connects
it to a free cell directly above, below, left, or right. `reachable_component()`
uses Breadth-First Search (BFS): start with a queue containing the candidate
head; remove the oldest queued cell, visit its unvisited free neighbours, and
append those neighbours. A visited set prevents infinite loops. When the queue
is empty, the visited set contains that whole connected region.

For a candidate first move, we first ask the actual collision predictor. If it
is illegal, reachable space is undefined (`null`). If legal, the old tail moves
away unless the known food is eaten. The resulting old body is then blocked,
while the candidate head is the BFS starting cell and counts as one reachable
cell. Future food placement is not generated. Candidate actions that share
the same obstacles reuse discovered components to avoid repeated search work.

After that first step, BFS freezes the body. It does not move the snake along
all paths, simulate future growth, or prove that a route can actually be followed
by a long moving body. More connected cells can be useful information, but they
are not a guaranteed measure of long-term safety.

### A concrete warning about static space

In seed 900045, move 152, candidate spaces were straight 384, right 1, left
384. The network chose right. Static BFS showed a closed one-cell region, yet
the next tail cell departed and provided a legal continuation. The episode
lasted another 1,239 moves. A focused unit test now reproduces that geometry.
Declaring a one-cell region to be certain death would have been wrong.

This does not prove the network made an optimal decision either: we did not
simulate the other policies. It shows why we need to distinguish an approximate
measurement from a claim about the consequences of a choice.

### What the full traces show

All 100 episodes again matched Milestones 4 and 5B, including scores, rewards,
lengths, and end flags. There are 63,728 pre-move states. The mean safe-action
count is 2.551469; 60.366% of movements had three safe choices, 34.572% had two,
4.905% had one, and 0.157% had none. The policy reduced next-state choices on
8.01% of legal transitions. That number alone says nothing about mistakes.

Before running, we defined low static space as at most 20 cells and a
substantial space deficit as at least 20 cells below another safe candidate.
There were 819 low-space moves, entered on 129 occasions, and 184 substantial
deficit moves. Those thresholds make the calculation explicit; they are not
claimed to be scientifically established safety cutoffs.

Selected area and remaining episode moves had a pooled Pearson correlation
of about 0.284. Pearson correlation summarises a linear association; it does
not demonstrate cause. Later states often have longer bodies and less time
remaining, and neighbouring observations are dependent. Longer episodes also
contribute more data. The analysis is descriptive, without a significance claim.

Near the end, options were narrower on average: the last-five-move mean was
1.186 safe actions. The final legal move entered static area one in all 100
episodes; 24 had a larger-area alternative, 23 with a gap of at least 20 cells.
We cannot claim those alternatives would survive because we did not play them
out. Earlier small regions sometimes reopened as the tail moved.

### Three examples, not proof of a general cause

We selected distinct examples using predefined descriptive roles. Seed 900034
had the highest score (58); at move 1360 it selected 12 static cells over a
328-cell alternative and collided 12 moves later. Seed 900045, also score 58,
had the most substantial deficits, including the one-cell example that later
escaped. Seed 900073 scored 57 and had 61 recoveries from one-safe runs; at
move 476 it selected 29 cells over 342 and survived another 818 moves.

These cases illustrate both narrowing and recovery. They do not prove that
the 11-feature observation caused a particular collision or that a new feature
would improve the model. The extra board information was recorded for analysis
only and never supplied to the policy.

### Why we did not add a three-move planner

The optional short lookahead was omitted to keep this milestone focused. Even
a three-move search would see only a short horizon and could miss deeper traps.
If a hypothetical branch eats known food, subsequent food placement is unknown;
a defensible lookahead would stop or mark uncertainty rather than invent a food
location. No amount of unqualified short search would guarantee safety.

### Saving full episodes economically

`state_traces.jsonl.gz` streams one compact JSON object per move through gzip.
It stores the full pre-state body once, along with the observation, Q-values,
chosen action, geometry, reward/end information and actual post head/food/score.
The full body is not duplicated in the CSVs. Small integer arrays hold plotting
measurements, and `trap_events.csv` logs selected events without repeating bodies.
The report documents nulls, coordinate conventions, units and remaining-time
joins. A failed run keeps an `INCOMPLETE` marker instead of a verified report.

The three Matplotlib charts show safe choices across complete example episodes,
selected reachable space, and comparisons near each example's largest observed
deficit. Markers keep isolated alternative values visible; missing collision
candidates are gaps, not fake zero areas.

### The next experiment remains a hypothesis

Future work could pre-register a controlled baseline versus a geometry-informed
observation design, with matched budgets, multiple training seeds, and fresh
untouched evaluation seeds. That would test whether the proposed information
actually helps. These already-inspected episodes should not be treated as a
fresh test set for tuning a new model.

The 18 focused tests and all 208 project tests pass. All 30 older saved files
remain unchanged. No model, reward, observation, algorithm, playback or training
implementation was changed, and no retraining or new agent was introduced.
Milestone 5C is verified for the checkpoint `feat: add SnakeAI trap formation analysis`.

## Milestone 5D: a controlled feature experiment

### Feature engineering is a hypothesis

A feature is a number that describes part of the environment to the model.
Our original observation gives 11 danger, direction, and food flags. It does
not tell the network how many cells remain before an obstacle. We now test
whether adding three relative static distances helps. We do not assume that
more information must improve learning: extra inputs can be redundant, hard to
use, or misleading. Static distance still ignores the body's future motion.

`SnakeEnv` now has an explicit observation mode. `original11` remains the default
and produces the same values as before. `distance14` appends the straight,
right, and left free-cell distances after all 11 original values. The geometry
calculation is shared with 5C through `game/geometry.py`; the basic environment
still does not import PyTorch, NumPy, or Pygame.

### Why normalise distances?

Danger flags are zero or one. A raw distance on a 20×20 board can be as large as
19, which would use a different numerical scale. We divide all three distances
by `max(width,height)-1`, a fixed board-dependent value. Thus four free cells
on this board give `4/19`, approximately 0.210526. Distances then fall between
zero and one without changing what counts as an obstacle.

The current tail is occupied for these static measurements, exactly as in 5C.
A distance of zero can therefore coexist with an immediate safe action when
the tail departs. The original danger flags continue to apply the real movement
rule. This apparent disagreement reflects two different questions, not a change
to collision rules.

### Keeping the comparison controlled

Model A has 11 inputs and 18,435 parameters; Model B has 14 inputs and 18,819.
Both have two 128-neuron ReLU hidden layers and three raw action values. The
extra 384 parameters are required by the larger input layer, not a separate
network-depth change. We reuse the same DQN loss, replay, Adam optimiser,
exploration schedule, and target updates for both.

The protocol was written before final training. Each run has a maximum of 500
episodes and 100,000 environment steps. Learning rate, gamma, rewards, episode
limits, batch size, warmup, and all other settings are fixed. If a technical
change is needed, it must be documented under a new protocol version rather
than quietly changing the question after seeing results.

The old seed-7 checkpoint is preserved and not used as a Model A replication.
Both variants train from scratch. Old checkpoints have no observation-mode
field; the loader interprets that as `original11`, keeping existing playback
and scripts compatible. New 14-feature checkpoints record their mode so the
loader can create a matching model and evaluation environment.

### Independent training seeds and pairing

The independent training seeds are 11, 22, and 33. Different random starts,
exploration choices, and replay batches can produce different learned policies.
Using three starts shows some of that variability. It is still a small sample
of possible training runs, so a consistent result would remain preliminary.

For a seed s, the action RNG uses s+2 and replay uses s+1. Python and PyTorch use
s. Both A and B begin episode i with food seed `200000 + 1000*s + i`. Matching
episode seeds makes the starting conditions comparable, while different choices
can lead to different later food sequences and episode lengths. The changed
input dimension also changes initialisation draws: the same seed does not mean
every hidden-layer weight is identical between architectures.

All six final checkpoints are tested greedily on environment seeds 910000
through 910099. These are separate from training, the earlier 900000-series
diagnostics, and the 920000-series smoke evaluation. Evaluation uses epsilon
zero and checks that weights and training state do not change. We do not select
the best seed or checkpoint using those outcomes.

### Validation versus scientific results

Before the six runs, focused tests check feature ordering, bounds, tail rules,
model/replay dimensions, legacy loading, reproducibility, and correct pairing.
A small smoke experiment uses development seed 77 in temporary directories.
Each mode must collect data, make finite updates, change weights, load its
checkpoint, and evaluate with the matching encoder without updating parameters.
Those technical smoke scores are not evidence for the research question.

The experiment runner freezes protocol, software versions, and source hashes
in the output directory. A completed run has an integrity marker, so running
the command again preserves it rather than retraining. Individual model/seed
selection lets us complete missing runs without repeating successful ones.
Incomplete training is preserved for review rather than approximately resumed
and presented as equivalent to an uninterrupted protocol run.

### What to average, and what variability means

Each trained model gets 100 evaluation games. We first calculate its mean,
median, variability, survival length, and end rates. Then we average the three
independent training-seed means for each variant. Their sample standard
deviation describes variability between learned policies, not uncertainty from
100 independent model-training replications.

For each training seed we subtract A's mean from B's mean. These three paired
differences show whether the direction is consistent and how large it is.
An average benefit driven by one very good run is different from a benefit
seen at all three starts. The 300 games per variant must not be treated as
300 independent training experiments. We use descriptive statistics without
an unnecessary significance test based on that false assumption.

Training charts include exploration, while evaluation is greedy, so their scores
need not match. Equal episode caps also allow unequal environment-step counts
and optimiser updates; the per-run results record both. Deterministic CPU
settings support reproduction here, but do not promise identical floating-point
behavior across operating systems or library versions. Final measurements and
the resulting cautious interpretation are recorded in the comparison report.

### What the six fixed runs actually showed

All six models completed 500 training episodes, with no budget extension.
A's evaluation means for training seeds 11, 22, and 33 were 29.70, 31.51, and
25.94. B's were 22.91, 13.31, and 23.67. Averaging each model's three seed means
gave A **29.05** and B **19.963333**. Their sample SDs across training-seed means
were **2.841320** and **5.774473**, respectively.

The paired B-minus-A differences were -6.79, -18.20, and -2.27, averaging
-9.086667. The direction was consistent: B scored lower on all three starts.
We therefore found no support for the proposed improvement under this fixed
budget and feature definition. We should preserve that result rather than
select the best B run or tune settings until a favourable comparison appears.
Three seeds still do not prove that the features are generally harmful.

A collected 33,282 / 35,500 / 33,742 training moves; B collected
30,903 / 34,005 / 30,170. Optimiser updates were those counts minus 999 because
learning began at move 1,000. Each model reached the episode cap before the
step cap or epsilon minimum. These different experience counts and remaining
exploration are part of the fixed-episode protocol, not identical computational
experience. A future equal-step experiment would ask a related but different
question and must be defined separately.

Secondary outcomes also deserve context. B/seed 22 had a longer mean episode
than A/seed 22, but a lower mean food score and 51% truncations. Remaining alive
until an episode limit is not the same as collecting food effectively. No
evaluated model filled the board. The saved per-episode records let readers
inspect these outcomes without reducing the comparison to a single best score.

Useful next work would be a new, explicitly controlled hypothesis about
representation or training budget, informed by these limitations and tested
with independent training seeds and fresh evaluation data. This milestone does
not implement that next experiment. All six final runs were completed once;
the temporary development smoke results were kept separate, and the original
seed-7 checkpoint and historical artifacts remain unchanged.

Final validation: 21 focused comparison tests passed, both development smoke
cycles passed, and the full suite ran once with 229 passing tests and no
failures (one dependency deprecation warning). Saved raw records reproduced
the reported metrics without repeating training or evaluation.

## Milestone 5E: relative food positions and matched budgets

The 11 original features say which direction contains food but not how far
away it is. `food13` adds horizontal and vertical displacement from the head.
On our 20-by-20 board, food five cells to the right and two cells above gives
dx=5/19 and dy=-2/19. Dividing by each board dimension minus one keeps values
between -1 and 1, including on rectangular boards. Coordinates increase
downward, so negative dy means above. These are world-axis offsets, not
offsets rotated into the snake's movement direction.

This is observation design: the agent receives more precision about something
already visible to a human. It receives no future food positions or rewards.
More information might help distinguish useful actions, but it also changes
the input layer and learning problem. The 5D results already showed why extra
features should be tested rather than assumed beneficial. Both original and
geometry encoders remain available, preserving previous checkpoint behaviour.

5D matched episode counts but produced different numbers of learning steps.
5E asks a new question with exactly 40,000 moves per model. Both agents reach
the same epsilon progress and update count; their experiences, completed
episodes and wall-clock times still differ. A final incomplete episode is
not counted as completed. We reuse the existing trainer and DQN algorithm.

Before training, we fixed seeds44/55/66, the food13 hypothesis and unchanged
hyperparameters in 5e-v1. A/B share an episode-food-seed schedule within each
pair. Different actions can still consume randomness differently, so paired
seeds are comparable initial conditions, not identical trajectories. The
same seed also does not create identical hidden weights across input sizes.

The requested evaluation seeds overlapped earlier smoke validation. We
identified that before training and obtained approval to reserve fresh
930000-930099 for final evaluation, with 940000-940004 reserved for smoke.
Final outcomes will not be used to adjust hyperparameters or select seeds.
We compare all three paired training runs and their variability, treating
three training seeds as exploratory evidence. Exact reproducibility across
software versions and operating systems is not guaranteed.

### What the measured 5E outcomes teach us

Both variants reached 40,000 interactions and 39,001 updates for every seed.
A's scores were 28.64, 25.94 and 24.21; B's were 27.41, 30.73 and 32.29.
The average increased from 26.26 to 30.14, but B lost the seed44 pair. That
supports a tentative average benefit, not a claim that food distance reliably
helps every training run. Its sample SD across seed means was slightly higher
(2.49 versus 2.23), so extra information did not reduce observed variability.

B improved median food score on all pairs, yet it did not consistently survive
longer. A/seed66 survived 1,023.21 moves on average with 37% truncations while
scoring less food. An agent can remain alive without efficiently pursuing food.
This is why our primary outcome is food score and secondary outcomes provide
context rather than replacing it after results are known.

Matched interactions also did not match completed episodes: counts ranged from
566 to 588. Both agents reached epsilon0.24, but different trajectories filled
their replay buffers with different experiences. Measured training times also
varied; matching update counts does not eliminate system-load differences.

The full suite ran once after implementation: 272 passed with no failures
and one dependency deprecation warning. Focused checks and both smoke cycles
passed before final training. Saved-data checks reproduced the metrics and
confirmed all 126 historical files unchanged without rerunning experiments.
The next experiment would need its own hypothesis, training seeds and fresh
evaluation seeds. We stop here, preserving the mixed outcomes for review.
