# SnakeAI: four-part LinkedIn series

Draft posts for manual review and publication. No content is posted automatically.

## Post 1 ? A reproducible environment before a learning algorithm

I built SnakeAI around a shared Python game engine: Pygame provides the human
interface, while a headless wrapper exposes observations, relative actions,
rewards and episode termination for reinforcement learning.

The important engineering choice was keeping those game rules identical.
Training, evaluation and visual playback all use the same transitions, with
explicit seeds and tests for collisions, rewards and reproducibility.

The original observation contains 11 features: immediate danger, heading and
food direction. Actions are straight, relative right and relative left.
A random-policy baseline provides a measured reference before interpreting
learned behaviour. Genuine terminal states and time-limit truncations are
handled separately when calculating learning targets.

This made the environment a testable experimental component rather than a
rendering loop that happened to contain an agent.

Suggested asset: architecture.svg plus a human-play screenshot.

#ReinforcementLearning #Python #MachineLearningEngineering

## Post 2 ? Training and inspecting a Deep Q-Network

SnakeAI's first DQN uses PyTorch with an 11?128?128?3 network, experience
replay, a target network, epsilon-greedy exploration and Huber-loss regression.

After 500 training episodes with one seed, greedy evaluation averaged 34.81
food items versus 0.14 for a random policy on the same 100 fresh environment
seeds. That is evidence of learning in this initial experiment, not a claim
of robust performance across independent training runs. Every evaluation
episode still ended in collision.

I added an inference interface showing the board, selected action, rewards
and all three Q estimates. Those values estimate future discounted reward;
they are not action probabilities. Pausing and replaying a fixed seed makes
individual decisions easier to inspect.

The useful outcome was a model whose behaviour could be examined alongside
its numerical evaluation, with checkpoints preserved separately from source.

Suggested asset: actual playback.gif and the initial evaluation panel.

#PyTorch #DeepLearning #ReinforcementLearning

## Post 3 ? Examining failure rather than stopping at the average score

The initial SnakeAI model could collect food, but every evaluated game ended
in collision. I used collision diagnostics and trap analysis to examine how.

At the final move of all 100 recorded collisions, there was no immediately safe
action. That does not mean the preceding policy was optimal: earlier choices
could leave the agent with too few options later.

The analysis tracked safe-action counts, connected free space and static
obstacle distances. Static reachable space could not guarantee survival
because the snake's body, especially its tail, moves. A constrained state
was also not necessarily fatal; many one-safe-action spells recovered.

Subsequent controlled feature experiments tested those ideas rather than
assuming more geometric information would help. Static obstacle and body
distances did not produce consistent performance improvements.

The lesson was to keep a diagnostic's interpretation narrower than the
behaviour we ultimately want: a geometric snapshot is not a planning model.

Suggested asset: safe_actions.png or the failure-analysis dashboard section.

#ModelEvaluation #ReinforcementLearning #DataScience

## Post 4 ? Controlled comparisons and the results that did not improve

I completed SnakeAI with a series of predefined comparisons, each using three
independent training seeds and paired held-out evaluations.

Food-distance inputs showed a tentative average benefit in their 40k-step
comparison. Adding static body distances did not consistently help. A separate
within-run study compared 40k, 60k, 80k and 100k snapshots: average score changed
from 30.00 to 30.19, with non-monotonic intermediate performance.

Finally, Standard DQN averaged 25.39 and Double DQN 25.71 under matched 100k
budgets. Double improved one of three seed pairs. Its observed seed-mean SD
was lower, but three seeds are insufficient to claim generally more stable
learning. An online-target disagreement diagnostic also did not establish
reduced overestimation bias.

These scores belong to different experimental contexts, not a universal
architecture ranking. The final portfolio preserves raw results, protocols,
checkpoints locally, an inspectable demo and a static evidence dashboard.

Negative and mixed results are part of the deliverable, not discarded runs.

Suggested asset: dashboard plus the within-run training-duration chart.

#MachineLearning #ExperimentDesign #PyTorch
