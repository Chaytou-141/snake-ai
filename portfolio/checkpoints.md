# Local checkpoint inventory

Weights are excluded from Git. This inventory describes the development machine;
`python -m snake_ai.demo --list-models` checks current local availability.
A clone does not receive these files, and no upload has been performed.

Selection rule: historical original, then first declared training seed/final
checkpoint per later variant. Scores are never used to rank or select models.

## original11
- Encoder: `original11`; algorithm: `dqn`.
- Selection: Historical original seed 7 final checkpoint.
- Local file: `results/milestone4_initial/checkpoints/final.pt`.
- Present at Version 1.0 verification: **True**.

## food13
- Encoder: `food13`; algorithm: `dqn`.
- Selection: First declared 5E training seed; final checkpoint.
- Local file: `results/milestone5e_food_distance/runs/B_seed44/training/checkpoints/final.pt`.
- Present at Version 1.0 verification: **True**.

## geometry14
- Encoder: `distance14`; algorithm: `dqn`.
- Selection: First declared 5D training seed; final checkpoint.
- Local file: `results/milestone5d_feature_comparison/runs/B_seed11/training/checkpoints/final.pt`.
- Present at Version 1.0 verification: **True**.

## body16
- Encoder: `body16`; algorithm: `dqn`.
- Selection: First declared 5F training seed; final checkpoint.
- Local file: `results/milestone5f_body_awareness/runs/B_seed111/training/checkpoints/final.pt`.
- Present at Version 1.0 verification: **True**.

## double_dqn
- Encoder: `food13`; algorithm: `double_dqn`.
- Selection: First declared 5H training seed; final checkpoint.
- Local file: `results/milestone5h_double_dqn/runs/B_seed244/training/checkpoints/final.pt`.
- Present at Version 1.0 verification: **True**.

The legacy playback default requires the original checkpoint. Each other catalog
entry requires only its own compatible file. Missing entries are not offered.
Custom files can be passed with `--checkpoint`. To obtain new weights explicitly,
see the Food13 example in README. No model is silently retrained.
