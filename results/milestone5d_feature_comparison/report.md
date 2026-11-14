# Milestone 5D — fixed feature comparison

## Protocol

Research question: does appending three static obstacle distances improve mean food score? The hypothesis was specified before training; benefit was not assumed. Protocol 5d-v1 uses training seeds 11/22/33, final checkpoints only, 500 episodes or 100,000 moves per run, and fresh evaluation seeds 910000–910099 shared by every model. No tuning or model selection.

A uses 11→128→128→3; B uses 14→128→128→3. The first 11 features are identical. B appends static free-cell distances straight/right/left divided by 19 on the 20×20 board. The tail counts as occupied for distance, even when immediate collision rules allow it to depart. Distances describe geometry, not future safety. Parameter counts are 18,435 and 18,819.

All other settings match: Adam 0.0005, gamma 0.99, replay 50,000, batch 64, warmup 1,000, target copy every 1,000 updates, gradient norm cap 10, epsilon 1→0.05 over 50,000 moves. Rewards and episode limits are unchanged. See protocol.json and metadata.json for complete settings, software and source hashes.

## Per-training-seed results

| Model | Training seed | Train episodes | Train moves | Updates | Eval mean | Eval median | Eval score SD | Mean eval length | Collision / truncation / win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 11 | 500 | 33282 | 32283 | 29.70 | 28.0 | 10.613 | 480.77 | 100% / 0% / 0% |
| B | 11 | 500 | 30903 | 29904 | 22.91 | 24.0 | 9.867 | 390.57 | 99% / 1% / 0% |
| A | 22 | 500 | 35500 | 34501 | 31.51 | 31.0 | 11.555 | 552.41 | 100% / 0% / 0% |
| B | 22 | 500 | 34005 | 33006 | 13.31 | 10.0 | 11.070 | 691.12 | 49% / 51% / 0% |
| A | 33 | 500 | 33742 | 32743 | 25.94 | 25.5 | 12.003 | 561.55 | 90% / 10% / 0% |
| B | 33 | 500 | 30170 | 29171 | 23.67 | 23.5 | 9.905 | 453.49 | 95% / 5% / 0% |

## Across independent training seeds

| Model | Mean of seed means | Sample SD across seed means |
| --- | ---: | ---: |
| A | 29.050000 | 2.841320 |
| B | 19.963333 | 5.774473 |

Paired B−A differences by training seed: {'11': -6.789999999999999, '22': -18.200000000000003, '33': -2.2699999999999996}. Mean paired difference: **-9.086667**. B higher on all three seeds: **False**. Direction consistently nonzero: **True**.

## Fairness and limitations

The independent training runs, not the 300 evaluation episodes per variant, are the replications. Three seeds provide preliminary descriptive evidence; no significance tests or broad superiority claim are made. The earlier seed-7 model and its already-used evaluation range were not part of this experiment.

For seed s, both variants use food seeds 200000+1000*s+episode index. Python/torch use s, replay s+1, exploration s+2. CPU uses one thread and deterministic algorithms. Same food seeds provide paired initial conditions, but different trajectories consume randomness differently. Input size changes initialisation draws, so equal seeds do not give identical hidden parameters. B necessarily adds 384 input-layer parameters. Equal episode caps allow unequal experience counts and epsilon progress; actual budgets are reported. Cross-platform/version determinism is not guaranteed.

Technical smoke validation uses seed 77 and evaluation seeds 920000–920004 in temporary directories; those results are not scientific findings. Each completed final run is hashed and never retrained on rerun. An interrupted but complete training phase can be reused for evaluation; incomplete training is preserved for review rather than silently restarted or approximately resumed.

## Files and reproduction

`runs/<model>_seed<seed>/training/` holds configuration, episode/update CSVs, summary, and ignored checkpoints. Each run's `evaluation.csv` records all 100 outcomes including model, training/evaluation seed, score, reward, moves, collision/truncation/win flags. `evaluation.json` stores metrics and hashes; `complete.json` records completion and artifact integrity. Root per_seed_summary.csv and comparison_summary.json contain the reported results. The four PNGs show per-seed mean scores, rolling training performance, episode lengths, and training-seed variability.

Run `python -m snake_ai.experiments.feature_comparison --smoke` after focused tests; then run the same module without --smoke. Use `--model A --seed 11` for one pair, preserving completed others. `--aggregate-only` reads completed runs without training. Existing root configuration/source hashes must match. Protocol changes require a new version/directory and documented reason before affected reruns.

Next work should be a separately approved, pre-registered experiment informed by variability and failures. Do not infer that more features always help or tune further on these now-inspected evaluation seeds.
