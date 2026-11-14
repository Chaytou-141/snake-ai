# SnakeAI — Milestone 5C trap formation analysis

## Reproduction and scope

All 100 episodes reproduced the saved Milestone 4 and 5B outcomes. Mean score 34.81, median 37.0, maximum 58, mean moves 637.28. No training or policy changes.

## Measurement definitions

Immediate safe actions use the original engine predictor and relative-action mapping. First occurrences mean exactly two, one, or zero safe actions before a one-based movement.

Obstacle distance counts **free cells** on a straight ray before a wall or current body. The current tail counts as a static obstacle for this geometric distance, even when the immediate move into it is legal because it departs. Distance zero therefore need not imply collision.

For each legal candidate first step, the old tail departs unless the known food is eaten. BFS then freezes the resulting body except the new head, counting its four-neighbour connected component **including that head cell**. Immediate collisions have null space, not zero. No future food is sampled and the real state/RNG is untouched. Static space ignores future tail movement, heading restrictions beyond the first step, and future growth; a larger area is not a guarantee.

Low space is <= 20 cells; a substantial deficit is >= 20 cells below the best immediately safe candidate. These thresholds were fixed before this run, not tuned to outcomes. The optional three-move search was deliberately not implemented.

## Immediate-option results

Recorded 63728 movements; mean safe-action count 2.551469.

| Safe actions | Movements | Percentage |
| --- | ---: | ---: |
| 0 | 100 | 0.157% |
| 1 | 3126 | 4.905% |
| 2 | 22032 | 34.572% |
| 3 | 38470 | 60.366% |

Mean static obstacle distance (free cells), over all recorded pre-states: straight 7.959515, right 7.468209, left 6.993347.

One-safe states recovered directly to two/three safe actions 1628 times out of 3126 one-safe states (52.079%). There were 1704 entries into consecutive one-safe runs and 76 one-to-zero transitions. 95.540% of these entered runs recovered to >=2 options. These are visits/runs, not independent experiments.

The chosen move reduced next-state immediate options in 5094 of 63628 nonterminal transitions (8.006%). Reduction is not labelled a mistake.

## Reachable space and remaining episode length

Low selected space occurred on 819 legal moves (1.287%), entered 129 times. The chosen action had a substantial deficit on 184 legal moves (0.289%).

Pooled Pearson r between selected space and remaining actual moves was 0.283731900202028. Remaining moves = total episode steps minus current move. Collision choices are excluded because their candidate area is undefined. Adjacent states are correlated, longer episodes receive more weight, and episode phase/body length confound this association. No causal interpretation, significance claim, or safety threshold is established.

| Selected reachable cells | Legal moves | Mean remaining moves |
| --- | ---: | ---: |
| 1-20 | 819 | 56.16727716727717 |
| 21-100 | 459 | 165.80392156862746 |
| 101+ | 62350 | 384.37177225340815 |

## Patterns near collisions

| Final window (including collision move) | States | Mean safe actions | Mean selected space on legal moves |
| --- | ---: | ---: | ---: |
| 1 | 100 | 0 | None |
| 5 | 500 | 1.186 | 83.025 |
| 10 | 1000 | 1.606 | 159.65333333333334 |
| 30 | 3000 | 2.1006666666666667 | 268.9303448275862 |

On the last legal move before collision, 100 of 100 episodes selected a candidate with static area 1. 24 had an immediately safe candidate with greater static space; 23 had a gap at least 20 cells. This is a local geometric comparison, not a counterfactual survival test.

The final lack of safe options reproduces 5B. Earlier narrow states can recover. Static region shrinkage can identify cases to inspect, but does not prove the selected action caused the eventual collision. The table describes aggregate windows, not a universal monotonic pattern.

## Three representative episodes

### highest_score: episode 35, seed 900034

Score 58, moves 1372; first two/one/zero-safe moves: 25 / 56 / 1372. One-safe recoveries 48; low-space entries 1; substantial space deficits 3. See the labelled plots and gzip trace for the actual states. These examples were selected by predefined descriptive roles, not as evidence that more features would necessarily improve the policy.

Largest observed deficit in this episode: move **1360**, head [7, 0] facing UP, chose **right**. Candidate areas (straight/right/left) were **[None, 12, 328]**; gap **316** cells. The actual episode continued **12** more moves. Next-state immediate safe count was **1**. In particular, a static area of 1 can still be escaped when the tail moves; small area is not proof of imminent death. No alternative policy was simulated.

### most_space_deficits: episode 46, seed 900045

Score 58, moves 1391; first two/one/zero-safe moves: 22 / 84 / 1391. One-safe recoveries 51; low-space entries 3; substantial space deficits 7. See the labelled plots and gzip trace for the actual states. These examples were selected by predefined descriptive roles, not as evidence that more features would necessarily improve the policy.

Largest observed deficit in this episode: move **152**, head [6, 13] facing UP, chose **right**. Candidate areas (straight/right/left) were **[384, 1, 384]**; gap **383** cells. The actual episode continued **1239** more moves. Next-state immediate safe count was **1**. In particular, a static area of 1 can still be escaped when the tail moves; small area is not proof of imminent death. No alternative policy was simulated.

### most_one_safe_recoveries: episode 74, seed 900073

Score 57, moves 1294; first two/one/zero-safe moves: 31 / 105 / 1294. One-safe recoveries 61; low-space entries 1; substantial space deficits 4. See the labelled plots and gzip trace for the actual states. These examples were selected by predefined descriptive roles, not as evidence that more features would necessarily improve the policy.

Largest observed deficit in this episode: move **476**, head [19, 10] facing RIGHT, chose **left**. Candidate areas (straight/right/left) were **[None, 342, 29]**; gap **313** cells. The actual episode continued **818** more moves. Next-state immediate safe count was **1**. In particular, a static area of 1 can still be escaped when the tail moves; small area is not proof of imminent death. No alternative policy was simulated.

## Files and schema

- `summary.json`: effective analysis/training environment configuration, seeds, hashes, reproduction checks, aggregate findings and example selection.
- `episodes.csv`: outcomes plus exact first safe-count occurrences, recoveries, option reductions, low-space entries/visits and deficit counts.
- `trap_events.csv`: only first-count occurrences, recovery/reduction/low-space/deficit events and episode ends; multiple labels separated by semicolons.
- `state_traces.jsonl.gz`: streamed UTF-8 JSON, one full pre-state body per move, with Q-values/observation, geometry and actual reward/end outcome. Post head/food/score are recorded without duplicating the full body. Coordinates are [column,row], body is head first, move/episode are one-based. Direction arrays follow straight/right/left. JSON null denotes an immediate-collision candidate or a terminal post-safe count. Join total episode steps from episodes.csv to calculate remaining moves. Scalar CSV nulls are blank.
- `safe_actions.png`, `reachable_space.png`, `action_comparison.png`: actual examples with episode/seed labels. The comparison chart focuses around each example's largest measured deficit. Missing candidate space is a gap, not a zero.

## Limitations and next controlled experiment

This reuses one frozen model and previously inspected seeds, so it is descriptive failure analysis, not new generalisation evidence. Geometry is not a dynamic Snake survival model. Three-move lookahead, even if added later, could miss traps deeper than its horizon and would need to stop or mark unknown continuation after hypothetical food collection rather than invent new food. Full board geometry here was never given to the network, and these results do not prove that observation aliasing caused a failure.

Next, pre-register a controlled baseline versus a geometry-informed observation experiment with matched budgets, multiple independent training seeds, and fresh untouched evaluation seeds. Keep reward/architecture choices explicit. Test whether a hypothesised improvement actually helps; no retraining, new agent, reward/feature redesign, or algorithm change is implemented here.
