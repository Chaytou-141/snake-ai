# Technical notes

## Learning system

The headless environment wraps the same grid engine as human play. Relative
actions are straight/right/left. Rewards are food +10, collision -10, ordinary
move 0, board-fill +10. Game termination is separate from budget truncation.
Standard and Double DQN bootstrap through truncation but not genuine terminals.

Both networks use two 128-unit ReLU hidden layers and three action-value outputs.
Observation modes are original11, food13, distance14 and body16. Food13 appends
signed normalised food offsets; body16 adds static body rays. Static tail
occupancy is documented separately from immediate tail-departure collision rules.

Online weights are fitted with Huber regression, Adam 0.0005, replay 50000,
batch 64 and clipping 10. Learning starts at 1000 moves; target weights copy every
1000 optimiser updates. Standard DQN selects/evaluates next actions with target;
Double selects with online and evaluates with target. Target calculations are
detached. Configurations default to Standard and preserve old checkpoint schemas.

## Experimental units and information boundaries

Model seeds, food schedules and evaluation ranges are frozen in configuration.
Each controlled comparison has three independent training seeds. Evaluation
episodes are paired environmental conditions, not independent training runs.
5D matched episode caps, not actual experience;5E/5F matched 40k moves;5H
matched 100k. 5G instead saved 40k/60k/80k/100k states along one continuous
trajectory per seed. Its snapshots are correlated repeated measurements.
No model was selected retrospectively from intermediate evaluation scores.

The 5H Q diagnostic uses a common 478-state pool sampled from required
evaluations of both algorithms. It measures online-target disagreement,
not return-based overestimation. Final targets lag one optimiser update.
Policy-dependent sampling and correlated states limit the diagnostic.

## Demonstration boundaries

Playback loads CPU weights, freezes gradients and uses inference_mode with
argmax actions. It never stores transitions, calls train_step or saves weights.
Model switching reconstructs the appropriate encoder/environment, replays the
same episode seed and pauses. Q-values refer to the current board; after
termination they retain the values before the final move, explicitly labelled.

Catalog selection uses the historical original plus the first declared seed's
final checkpoint per later variant, independent of held-out performance.
Only compatible local files are offered. Checkpoints remain excluded from Git;
no release, LFS or third-party upload is performed. A clone can play human
Snake and view the recorded dashboard without weights.

## Remaining limits

Three training seeds are exploratory evidence. No result proves universal
feature/algorithm superiority or statistical equivalence. Static geometry does
not model future body motion. Every final 5H evaluation ended in collision.
Training duration under a fixed epsilon schedule also changes experienced
exploration, so duration and exploration effects are not separately identified.
Exact cross-platform/version reproducibility is not guaranteed. Saved inference
checkpoints omit active-game and replay state, preventing exact resume.
