# SnakeAI Version 1.0

SnakeAI is a Python/Pygame and PyTorch portfolio project exploring learned
control, observation design and reproducible reinforcement-learning experiments.
One game engine supports human play, a headless environment, Standard DQN,
Double DQN, held-out evaluation and an inference interface with live action values.

The initial single-seed 11-feature DQN averaged 34.81 food items versus 0.14 for
a random policy on the same 100 evaluation seeds. All games still ended in
collision. Controlled studies then examined obstacle distances, food offsets,
body proximity, training duration and Double DQN across three training seeds.
Food13 was promising in its comparison; further feature and algorithm changes
did not deliver consistent gains. Different studies are not a universal ranking.

The implementation separates game rules, encoding, replay, network updates,
evaluation and rendering. Protocols, raw outcomes, source/software hashes and
negative results are preserved. Default checkpoint compatibility is maintained.
The local demo offers only available compatible weights; inference never trains.
The static dashboard checks saved aggregates and retains each study's context.

## Deliverables

- [Demonstration](../README.md#local-demonstration), [real playback clip](assets/playback.gif).
- [Architecture](architecture.svg), [maintainable source](architecture.mmd).
- [Static evidence dashboard](dashboard/index.html).
- [Technical notes](technical_notes.md), [checkpoint inventory](checkpoints.md).
- [LinkedIn series](linkedin_series.md), [recording guide](recording_guide.md).
- [Final verification](verification.md).

## Portfolio summary

Implemented a reproducible DQN-based Snake control system with configurable
observations, Double DQN targets, controlled paired experiments, failure
analysis and a local inference interface. Preserved measured negative outcomes
and experimental limitations alongside learning improvements.
