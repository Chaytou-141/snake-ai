# SnakeAI — Milestone 5B collision diagnostics

## Evaluation overview

Greedy inference on 100 recorded evaluation seeds. Every episode's score, moves, reward, end flags, and end reason matched the original Milestone 4 CSV; all summary metrics matched.

Mean score **34.81**, median **37.0**, maximum **58**, mean moves **637.28**. Collision rate **100%**. No training or policy changes.

## Collision classification

| Category | Count | Percentage of collisions |
| --- | ---: | ---: |
| A — Avoidable immediate collision | 0 | 0.0% |
| B — No immediately safe action | 100 | 100.0% |
| C — Other or uncertain | 0 | 0.0% |

Type A means a safe *next move* existed; it does not show that taking it would avoid a later trap. Type B means no safe next move existed at the final state; it does not prove earlier decisions could not have prevented reaching that state. Type C covers inconsistent evidence or other endings. Win/truncation endings are reported separately from collision percentages.

## Q-value analysis

The selected action was immediately dangerous in **100** collisions. All selections used the largest raw Q-value (ties choose the first action).
Selected collision Q-values averaged **-9.417852**, range **-10.043816 to -8.523903**. 0 were nonnegative. These are predictions to inspect, not a probability or an independent safety classifier.
No Type A collisions were observed, so there is no selected-versus-safe Q gap to estimate.

Q-values estimate discounted future reward, not probabilities or immediate collision risk. A negative value is not an independent safety test; safety here comes only from the engine predictor.

## Failure examples

### Safe alternative available

No matching event occurred in this sample; an example cannot be supplied.

### No safe next action

Episode **1**, seed **900000**, score **40**, move **838**, category **B**. Head [1, 8], facing DOWN; body length 43.

| Action | Q-value | Immediate collision |
| --- | ---: | --- |
| straight | -10.175884 | True |
| right (selected) | -9.489347 | True |
| left | -10.461297 | True |

Immediately safe alternatives: **none**. Attempted destination [0, 8]; actual post-action head [1, 8].
This supports the immediate option comparison only. It does not identify the cause of earlier decisions, prove a safe alternative would survive, or establish that missing board information caused this collision.

### Highest score before collision

Episode **35**, seed **900034**, score **58**, move **1372**, category **B**. Head [19, 0], facing RIGHT; body length 61.

| Action | Q-value | Immediate collision |
| --- | ---: | --- |
| straight | -10.010465 | True |
| right (selected) | -9.271073 | True |
| left | -10.779609 | True |

Immediately safe alternatives: **none**. Attempted destination [19, 1]; actual post-action head [19, 0].
This supports the immediate option comparison only. It does not identify the cause of earlier decisions, prove a safe alternative would survive, or establish that missing board information caused this collision.

Examples select the first observed A and B events and the maximum-score collision (first on ties); roles may overlap. Full body coordinates and final 30 moves are in `example_failure.json`.

## Schema and reading the files

- `episodes.csv`: episode, seed, final score, steps, total reward, end flags/reason, category/note, retained count.
- `final_moves.csv`: at most 30 actual moves per episode. `*_before`/`*_after` fields give head, direction, food, length, score, and complete head-first body. `observation` follows the existing 11-feature order recorded in summary.json. `q_straight/right/left`, `selected_action` (0/1/2), `action_name`, and actual `movement_direction` describe the decision. `danger_*` are pre-action engine predictions. `safe_actions`, best-safe values, Q gap, reward, end flags/reason, and consistency flags describe the outcome. `attempted_head` is a projected destination, not the actual post-state.
- `collisions.csv`: the final move of each collision with category/note. Same move schema as above.
- Coordinates, bodies, observations, and safe-action lists are JSON encoded inside CSV cells; absent food is JSON `null`. CSV booleans are `True`/`False`, nullable scalar fields are blank, and move/episode numbers are one-based. Coordinates are (column, row), origin top-left. A collision leaves the actual body/head in its last valid position, per the existing engine.
- `summary.json`: configuration, hashes, exact seeds, reproduction status, statistics and preservation checks.
- `example_failure.json`: three named example roles, each containing an episode row, final move, and retained moves; a missing category has JSON `null`.

## Limitations and next experiments

This is one already-trained model and 100 previously evaluated seeds, not fresh generalisation evidence. The 11 inputs omit the full board/body geometry, distances, and budgets. Immediate safety cannot reveal long-term trapping; imperfect Q estimates can also rank actions poorly. These are possible mechanisms, not demonstrated causes for individual failures. Only the last 30 moves are retained, so earlier setup may be missing. The extra diagnostic body data was never supplied to the policy.

Next, inspect earlier trap formation with explicitly defined geometry/planning measurements, then pre-register any observation or algorithm comparison and evaluate multiple independent training seeds on fresh untouched seeds. Type A failures motivate value-ranking/calibration checks; Type B failures motivate investigation of earlier planning. This run implements no improvements or training experiments.
