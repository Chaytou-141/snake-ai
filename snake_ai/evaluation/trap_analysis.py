"""Read-only geometry diagnostics over full greedy episodes; no planning agent."""

import argparse
from array import array
from collections import Counter, deque
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.game.geometry import free_cell_distance
from snake_ai.evaluation.collision_diagnostics import (
    ACTION_NAMES, DEFAULT_CHECKPOINT, DEFAULT_REFERENCE, record_move, verify_reproduction,
)
from snake_ai.rl import SnakeEnv, relative_direction
from snake_ai.training.evaluate import summarise

DEFAULT_OUTPUT = Path("results/milestone5c_trap_analysis")
DEFAULT_5B = Path("results/milestone5b_collision_diagnostics")
EVENT_FIELDS = ["episode", "environment_seed", "move", "events", "safe_before", "safe_after",
                "selected_action", "selected_space", "best_safe_space", "space_deficit", "score"]
HISTORY_FIELDS = ("safe_before", "safe_after", "selected_space", "best_space", "straight", "right", "left")


def reachable_component(start: tuple[int, int], blocked: set[tuple[int, int]],
                        width: int, height: int) -> set[tuple[int, int]]:
    """Four-neighbour BFS, counting the start cell; no body movement thereafter."""
    if start in blocked or not (0 <= start[0] < width and 0 <= start[1] < height):
        return set()
    visited = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for point in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= point[0] < width and 0 <= point[1] < height
                    and point not in blocked and point not in visited):
                visited.add(point)
                queue.append(point)
    return visited


def measure_state(game) -> dict:
    """Candidate first step follows the engine predictor and eating/tail rules.

    After a legal move, the obstacles are the old snake without its departing
    tail, or the entire old snake when eating. The new head is the BFS start,
    not an obstacle. No future food is generated or hypothetical RNG consumed.
    """
    directions = [relative_direction(game.direction, action) for action in range(3)]
    danger = [game.would_collide(direction) for direction in directions]
    distances = [free_cell_distance(game, direction) for direction in directions]
    spaces: list[int | None] = []
    # Candidates with the same growth state share obstacles. Reuse a component
    # once discovered, instead of repeating identical BFS work for each action.
    caches: dict[bool, dict[tuple[int, int], int]] = {}
    for action, direction in enumerate(directions):
        if danger[action]:
            spaces.append(None)
            continue
        dx, dy = direction.value
        x, y = game.snake[0]
        target = (x + dx, y + dy)
        growing = target == game.food
        cache = caches.setdefault(growing, {})
        if target not in cache:
            blocked = set(game.snake if growing else game.snake[:-1])
            component = reachable_component(target, blocked, game.width, game.height)
            cache.update(dict.fromkeys(component, len(component)))
        spaces.append(cache[target])
    return {"safe_action_count": 3 - sum(danger), "danger": danger,
            "obstacle_free_cells": distances, "reachable_cells": spaces}


def _optional(value):
    return None if value == -1 else value


def analyse_episode(env: SnakeEnv, agent: DQNAgent, episode: int, seed: int,
                    trace, event_writer, *, low_space_cells: int, substantial_gap_cells: int) -> tuple[dict, dict]:
    env.reset(seed=seed)
    history = {name: array("i") for name in HISTORY_FIELDS}
    first = {count: None for count in (2, 1, 0)}
    previous_safe = None
    previous_low = False
    total_reward = 0.0
    one_entries = recoveries = one_to_zero = reductions = low_moves = low_entries = deficits = 0
    while True:
        geometry = measure_state(env.game)
        row = record_move(env, agent, episode, seed)
        safe = geometry["safe_action_count"]
        if geometry["danger"] != [bool(value) for value in row["observation"][:3]]:
            raise RuntimeError("Safety measurements disagree with the existing observation encoder")
        post_safe = None if row["terminated"] else 3 - sum(
            env.game.would_collide(relative_direction(env.game.direction, action)) for action in range(3))
        chosen = geometry["reachable_cells"][row["selected_action"]]
        legal_spaces = [value for value in geometry["reachable_cells"] if value is not None]
        best = max(legal_spaces) if legal_spaces else None
        deficit = best - chosen if chosen is not None else None
        low = chosen is not None and chosen <= low_space_cells
        events = []
        if safe in first and first[safe] is None:
            first[safe] = row["move"]
            events.append(f"first_{safe}_safe")
        if safe == 1 and previous_safe != 1:
            one_entries += 1
        if safe == 1 and post_safe is not None and post_safe >= 2:
            recoveries += 1
            events.append("one_safe_recovery")
        if safe == 1 and post_safe == 0:
            one_to_zero += 1
        if post_safe is not None and post_safe < safe:
            reductions += 1
            events.append("options_reduced")
        if low:
            low_moves += 1
            if not previous_low:
                low_entries += 1
                events.append("entered_low_space")
        if deficit is not None and deficit >= substantial_gap_cells:
            deficits += 1
            events.append("substantial_space_deficit")
        if row["terminated"] or row["truncated"]:
            events.append("episode_end")
        if events:
            event_writer.writerow({"episode": episode, "environment_seed": seed, "move": row["move"],
                "events": ";".join(events), "safe_before": safe, "safe_after": post_safe,
                "selected_action": row["selected_action"], "selected_space": chosen,
                "best_safe_space": best, "space_deficit": deficit, "score": row["score_before"]})
        # One pre-state body per move. Scalars plus the terminal post-state are
        # sufficient; avoid duplicating full bodies in a second CSV or after-state.
        state = {key: row[key] for key in ("episode", "environment_seed", "move", "head_before", "body_before",
            "food_before", "direction_before", "score_before", "snake_length_before", "observation",
            "q_straight", "q_right", "q_left", "selected_action", "selected_q", "movement_direction",
            "reward", "terminated", "truncated", "end_reason", "head_after", "food_after", "score_after")}
        state.update(geometry)
        state.update(safe_action_count_after=post_safe, selected_reachable_cells=chosen,
                     best_safe_reachable_cells=best, space_deficit=deficit)
        trace.write(json.dumps(state, separators=(",", ":")) + "\n")
        values = (safe, post_safe, chosen, best, *geometry["reachable_cells"])
        for name, value in zip(HISTORY_FIELDS, values):
            history[name].append(-1 if value is None else value)
        total_reward += row["reward"]
        previous_safe, previous_low = safe, low
        if row["terminated"] or row["truncated"]:
            break
    result = {"episode": episode, "environment_seed": seed, "score": row["score_after"], "steps": row["move"],
        "total_reward": total_reward, "terminated": row["terminated"], "truncated": row["truncated"],
        "end_reason": row["end_reason"], "first_two_safe_move": first[2], "first_one_safe_move": first[1],
        "first_zero_safe_move": first[0], "one_safe_spell_entries": one_entries,
        "one_safe_recoveries": recoveries, "one_safe_to_zero": one_to_zero,
        "option_reducing_moves": reductions, "low_space_moves": low_moves,
        "low_space_entries": low_entries, "substantial_deficit_moves": deficits}
    return result, history


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    n = len(pairs)
    sx, sy = sum(x for x, _ in pairs), sum(y for _, y in pairs)
    xx, yy = sum(x*x for x, _ in pairs), sum(y*y for _, y in pairs)
    xy = sum(x*y for x, y in pairs)
    denominator = (n*xx - sx*sx) * (n*yy - sy*sy)
    return (n*xy - sx*sy) / math.sqrt(denominator) if denominator > 0 else None


def aggregate(episodes: list[dict], histories: list[dict], low_space_cells: int, gap_cells: int) -> dict:
    counts = Counter(value for history in histories for value in history["safe_before"])
    moves = sum(counts.values())
    legal = sum(value >= 0 for history in histories for value in history["selected_space"])
    transitions = sum(value >= 0 for history in histories for value in history["safe_after"])
    pairs = [(space, len(history["selected_space"]) - index - 1) for history in histories
             for index, space in enumerate(history["selected_space"]) if space >= 0]
    bins = []
    for name, low, high in (("1-20", 1, 20), ("21-100", 21, 100), ("101+", 101, float("inf"))):
        group = [remaining for space, remaining in pairs if low <= space <= high]
        bins.append({"reachable_cells": name, "moves": len(group),
                     "mean_remaining_moves": statistics.mean(group) if group else None})
    windows = []
    collision_histories = [history for episode, history in zip(episodes, histories)
                           if episode["end_reason"] == "collision"]
    for size in (1, 5, 10, 30):
        safes = [value for history in collision_histories for value in history["safe_before"][-size:]]
        spaces = [value for history in collision_histories for value in history["selected_space"][-size:] if value >= 0]
        windows.append({"last_moves_including_collision": size, "recorded_states": len(safes),
            "mean_safe_actions": statistics.mean(safes) if safes else None,
            "mean_selected_space_legal_moves": statistics.mean(spaces) if spaces else None})
    penultimate = [history for history in collision_histories if len(history["safe_before"]) >= 2]
    one_rows = counts[1]
    one_entries = sum(row["one_safe_spell_entries"] for row in episodes)
    reductions = sum(row["option_reducing_moves"] for row in episodes)
    recoveries = sum(row["one_safe_recoveries"] for row in episodes)
    return {"movement_count": moves, "legal_chosen_moves": legal, "mean_safe_actions":
        sum(count*frequency for count, frequency in counts.items()) / moves,
        "safe_action_counts": {str(count): {"count": counts[count], "percentage": 100*counts[count]/moves}
                               for count in range(4)},
        "one_safe_states": one_rows, "one_safe_spell_entries": one_entries,
        "one_safe_recoveries_to_two_or_three": recoveries,
        "one_safe_recovery_percentage_of_one_safe_states": 100*recoveries/one_rows if one_rows else 0.,
        "one_safe_spell_recovery_percentage": 100*recoveries/one_entries if one_entries else 0.,
        "one_safe_to_zero": sum(row["one_safe_to_zero"] for row in episodes),
        "option_reducing_moves": reductions, "nonterminal_transitions_measured": transitions,
        "option_reducing_percentage": 100*reductions/transitions if transitions else 0.,
        "low_space_cells_threshold": low_space_cells,
        "low_space_moves": sum(row["low_space_moves"] for row in episodes),
        "low_space_percentage_of_legal_moves": 100*sum(row["low_space_moves"] for row in episodes)/legal if legal else 0.,
        "low_space_entries": sum(row["low_space_entries"] for row in episodes),
        "substantial_gap_cells_threshold": gap_cells,
        "substantial_deficit_moves": sum(row["substantial_deficit_moves"] for row in episodes),
        "substantial_deficit_percentage_of_legal_moves": 100*sum(row["substantial_deficit_moves"] for row in episodes)/legal if legal else 0.,
        "reachable_space_remaining_moves_pearson_r": _correlation(pairs),
        "remaining_moves_by_space_bin": bins, "pre_collision_windows": windows,
        "last_legal_move_before_collision": {
            "episodes": len(penultimate),
            "selected_static_space_one": sum(history["selected_space"][-2] == 1 for history in penultimate),
            "larger_space_alternative": sum(history["best_space"][-2] > history["selected_space"][-2]
                                            for history in penultimate),
            "substantial_space_alternative": sum(history["best_space"][-2] - history["selected_space"][-2] >= gap_cells
                                                 for history in penultimate)}}


def _examples(episodes: list[dict]) -> list[dict]:
    """Three distinct, reproducibly selected roles where episode count permits."""
    selections, used = [], set()
    rankings = (("highest_score", lambda row: row["score"]),
                ("most_space_deficits", lambda row: row["substantial_deficit_moves"]),
                ("most_one_safe_recoveries", lambda row: row["one_safe_recoveries"]))
    for role, key in rankings:
        candidates = [row for row in episodes if row["episode"] not in used]
        if candidates:
            selected = max(candidates, key=key)
            used.add(selected["episode"])
            selections.append({"role": role, **selected})
    return selections


def decision_example(row: dict, episode_steps: int) -> dict:
    """Compact pointer to an observed decision; full body stays in gzip trace."""
    return {key: row[key] for key in ("move", "head_before", "direction_before", "score_before",
        "safe_action_count", "safe_action_count_after", "selected_action", "selected_q",
        "q_straight", "q_right", "q_left", "reachable_cells", "space_deficit")} | {
            "remaining_moves_after_action": episode_steps - row["move"]}


def create_charts(output: Path, examples: list[dict], histories: list[dict], low_space_cells: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for chart, title in (("safe_actions", "Immediate safe actions before each move"),
                         ("reachable_space", "Selected action: static reachable cells after first move"),
                         ("action_comparison", "Candidate space near each example's largest measured deficit")):
        fig, axes = plt.subplots(len(examples), 1, figsize=(11, 3*len(examples)), squeeze=False, constrained_layout=True)
        fig.suptitle(title)
        for axis, example in zip(axes.flat, examples):
            history = histories[example["episode"] - 1]
            count = len(history["safe_before"])
            if chart == "safe_actions":
                axis.step(range(1, count+1), history["safe_before"], where="post")
                axis.set(ylim=(-0.15, 3.2), yticks=[0, 1, 2, 3], ylabel="Safe actions")
            elif chart == "reachable_space":
                axis.plot(range(1, count+1), [_optional(value) for value in history["selected_space"]])
                axis.axhline(low_space_cells, linestyle="--", color="gray", label=f"{low_space_cells}-cell reference")
                axis.set(ylabel="Reachable cells")
            else:
                anchor = example.get("largest_deficit_decision", {}).get("move", count)
                start, stop = max(0, anchor - 30), min(count, anchor + 30)
                for name in ACTION_NAMES:
                    axis.plot(range(start+1, stop+1), [_optional(value) for value in history[name][start:stop]],
                              label=name, linewidth=1, marker=".", markersize=4)
                axis.plot(range(start+1, stop+1), [_optional(value) for value in history["selected_space"][start:stop]],
                          color="black", linestyle="--", label="selected", linewidth=1.2, marker=".", markersize=4)
                axis.axvline(anchor, color="gray", linestyle=":", linewidth=0.7)
                axis.set(ylabel="Reachable cells")
                axis.legend(loc="upper right", fontsize=8)
            axis.set(title=f"{example['role'].replace('_', ' ')} | episode {example['episode']}, seed {example['environment_seed']}, score {example['score']}",
                     xlabel="Move (one-based)")
            axis.grid(alpha=0.2)
        fig.savefig(output / f"{chart}.png", dpi=150)
        plt.close(fig)


def write_report(output: Path, summary: dict) -> None:
    metrics, analysis = summary["evaluation"], summary["analysis"]
    lines = ["# SnakeAI — Milestone 5C trap formation analysis", "", "## Reproduction and scope", "",
        f"All {metrics['episodes']} episodes reproduced the saved Milestone 4 and 5B outcomes. "
        f"Mean score {metrics['average_score']}, median {metrics['median_score']}, maximum {metrics['maximum_score']}, "
        f"mean moves {metrics['average_episode_length']}. No training or policy changes.", "",
        "## Measurement definitions", "",
        "Immediate safe actions use the original engine predictor and relative-action mapping. "
        "First occurrences mean exactly two, one, or zero safe actions before a one-based movement.", "",
        "Obstacle distance counts **free cells** on a straight ray before a wall or current body. "
        "The current tail counts as a static obstacle for this geometric distance, even when the immediate "
        "move into it is legal because it departs. Distance zero therefore need not imply collision.", "",
        "For each legal candidate first step, the old tail departs unless the known food is eaten. "
        "BFS then freezes the resulting body except the new head, counting its four-neighbour connected "
        "component **including that head cell**. Immediate collisions have null space, not zero. "
        "No future food is sampled and the real state/RNG is untouched. Static space ignores future tail "
        "movement, heading restrictions beyond the first step, and future growth; a larger area is not a guarantee.", "",
        f"Low space is <= {analysis['low_space_cells_threshold']} cells; a substantial deficit is >= "
        f"{analysis['substantial_gap_cells_threshold']} cells below the best immediately safe candidate. "
        "These thresholds were fixed before this run, not tuned to outcomes. The optional three-move search "
        "was deliberately not implemented.", "", "## Immediate-option results", "",
        f"Recorded {analysis['movement_count']} movements; mean safe-action count {analysis['mean_safe_actions']:.6f}.", "",
        "| Safe actions | Movements | Percentage |", "| --- | ---: | ---: |"]
    for count, item in analysis["safe_action_counts"].items():
        lines.append(f"| {count} | {item['count']} | {item['percentage']:.3f}% |")
    lines += ["", "Mean static obstacle distance (free cells), over all recorded pre-states: " +
              ", ".join(f"{name} {value:.6f}" for name, value in summary["mean_obstacle_free_cells"].items()) + "."]
    lines += ["", f"One-safe states recovered directly to two/three safe actions {analysis['one_safe_recoveries_to_two_or_three']} "
        f"times out of {analysis['one_safe_states']} one-safe states "
        f"({analysis['one_safe_recovery_percentage_of_one_safe_states']:.3f}%). "
        f"There were {analysis['one_safe_spell_entries']} entries into consecutive one-safe runs and "
        f"{analysis['one_safe_to_zero']} one-to-zero transitions. "
        f"{analysis['one_safe_spell_recovery_percentage']:.3f}% of these entered runs recovered to >=2 options. "
        "These are visits/runs, not independent experiments.", "",
        f"The chosen move reduced next-state immediate options in {analysis['option_reducing_moves']} of "
        f"{analysis['nonterminal_transitions_measured']} nonterminal transitions "
        f"({analysis['option_reducing_percentage']:.3f}%). Reduction is not labelled a mistake.", "",
        "## Reachable space and remaining episode length", "",
        f"Low selected space occurred on {analysis['low_space_moves']} legal moves "
        f"({analysis['low_space_percentage_of_legal_moves']:.3f}%), entered {analysis['low_space_entries']} times. "
        f"The chosen action had a substantial deficit on {analysis['substantial_deficit_moves']} legal moves "
        f"({analysis['substantial_deficit_percentage_of_legal_moves']:.3f}%).", "",
        f"Pooled Pearson r between selected space and remaining actual moves was "
        f"{analysis['reachable_space_remaining_moves_pearson_r']}. Remaining moves = total episode steps minus current move. "
        "Collision choices are excluded because their candidate area is undefined. Adjacent states are correlated, "
        "longer episodes receive more weight, and episode phase/body length confound this association. No causal "
        "interpretation, significance claim, or safety threshold is established.", "",
        "| Selected reachable cells | Legal moves | Mean remaining moves |", "| --- | ---: | ---: |"]
    for item in analysis["remaining_moves_by_space_bin"]:
        lines.append(f"| {item['reachable_cells']} | {item['moves']} | {item['mean_remaining_moves']} |")
    lines += ["", "## Patterns near collisions", "",
        "| Final window (including collision move) | States | Mean safe actions | Mean selected space on legal moves |",
        "| --- | ---: | ---: | ---: |"]
    for item in analysis["pre_collision_windows"]:
        lines.append(f"| {item['last_moves_including_collision']} | {item['recorded_states']} | "
            f"{item['mean_safe_actions']} | {item['mean_selected_space_legal_moves']} |")
    last = analysis["last_legal_move_before_collision"]
    lines += ["", f"On the last legal move before collision, {last['selected_static_space_one']} of "
        f"{last['episodes']} episodes selected a candidate with static area 1. "
        f"{last['larger_space_alternative']} had an immediately safe candidate with greater static space; "
        f"{last['substantial_space_alternative']} had a gap at least {analysis['substantial_gap_cells_threshold']} cells. "
        "This is a local geometric comparison, not a counterfactual survival test."]
    lines += ["", "The final lack of safe options reproduces 5B. Earlier narrow states can recover. "
        "Static region shrinkage can identify cases to inspect, but does not prove the selected action caused "
        "the eventual collision. The table describes aggregate windows, not a universal monotonic pattern.", "",
        "## Three representative episodes", ""]
    for example in summary["examples"]:
        lines += [f"### {example['role']}: episode {example['episode']}, seed {example['environment_seed']}", "",
            f"Score {example['score']}, moves {example['steps']}; first two/one/zero-safe moves: "
            f"{example['first_two_safe_move']} / {example['first_one_safe_move']} / {example['first_zero_safe_move']}. "
            f"One-safe recoveries {example['one_safe_recoveries']}; low-space entries {example['low_space_entries']}; "
            f"substantial space deficits {example['substantial_deficit_moves']}. "
            "See the labelled plots and gzip trace for the actual states. These examples were selected by "
            "predefined descriptive roles, not as evidence that more features would necessarily improve the policy.", ""]
        decision = example.get("largest_deficit_decision")
        if decision is not None:
            lines += [f"Largest observed deficit in this episode: move **{decision['move']}**, "
                f"head {decision['head_before']} facing {decision['direction_before']}, chose "
                f"**{ACTION_NAMES[decision['selected_action']]}**. Candidate areas (straight/right/left) "
                f"were **{decision['reachable_cells']}**; gap **{decision['space_deficit']}** cells. "
                f"The actual episode continued **{decision['remaining_moves_after_action']}** more moves. "
                f"Next-state immediate safe count was **{decision['safe_action_count_after']}**. "
                "In particular, a static area of 1 can still be escaped when the tail moves; small area is not "
                "proof of imminent death. No alternative policy was simulated.", ""]
    lines += ["## Files and schema", "",
        "- `summary.json`: effective analysis/training environment configuration, seeds, hashes, reproduction checks, aggregate findings and example selection.",
        "- `episodes.csv`: outcomes plus exact first safe-count occurrences, recoveries, option reductions, low-space entries/visits and deficit counts.",
        "- `trap_events.csv`: only first-count occurrences, recovery/reduction/low-space/deficit events and episode ends; multiple labels separated by semicolons.",
        "- `state_traces.jsonl.gz`: streamed UTF-8 JSON, one full pre-state body per move, with Q-values/observation, geometry and actual reward/end outcome. "
        "Post head/food/score are recorded without duplicating the full body. Coordinates are [column,row], body is head first, move/episode are one-based. "
        "Direction arrays follow straight/right/left. JSON null denotes an immediate-collision candidate or a terminal post-safe count. "
        "Join total episode steps from episodes.csv to calculate remaining moves. Scalar CSV nulls are blank.",
        "- `safe_actions.png`, `reachable_space.png`, `action_comparison.png`: actual examples with episode/seed labels. The comparison chart focuses around each example's largest measured deficit. Missing candidate space is a gap, not a zero.", "",
        "## Limitations and next controlled experiment", "",
        "This reuses one frozen model and previously inspected seeds, so it is descriptive failure analysis, "
        "not new generalisation evidence. Geometry is not a dynamic Snake survival model. Three-move lookahead, "
        "even if added later, could miss traps deeper than its horizon and would need to stop or mark unknown "
        "continuation after hypothetical food collection rather than invent new food. Full board geometry here "
        "was never given to the network, and these results do not prove that observation aliasing caused a failure.", "",
        "Next, pre-register a controlled baseline versus a geometry-informed observation experiment with "
        "matched budgets, multiple independent training seeds, and fresh untouched evaluation seeds. Keep "
        "reward/architecture choices explicit. Test whether a hypothesised improvement actually helps; no "
        "retraining, new agent, reward/feature redesign, or algorithm change is implemented here.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(checkpoint: Path = DEFAULT_CHECKPOINT, output: Path = DEFAULT_OUTPUT,
                 reference: Path = DEFAULT_REFERENCE, reference_5b: Path = DEFAULT_5B, *,
                 low_space_cells: int = 20, substantial_gap_cells: int = 20, make_charts: bool = True) -> dict:
    checkpoint, output, reference, reference_5b = map(Path, (checkpoint, output, reference, reference_5b))
    if output.exists():
        raise FileExistsError(f"Choose a new analysis directory: {output}")
    if any(type(value) is not int or value < 1 for value in (low_space_cells, substantial_gap_cells)):
        raise ValueError("Analysis thresholds must be positive integers")
    protected = [checkpoint, reference/"summary.json", reference/"episodes.csv",
                 reference_5b/"summary.json", reference_5b/"episodes.csv"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    ref = json.loads((reference/"summary.json").read_text())
    ref5b = json.loads((reference_5b/"summary.json").read_text())
    if hashes[str(checkpoint)] != ref["checkpoint_sha256"] or ref5b["checkpoint_sha256"] != ref["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match both verified reference hashes")
    agent = DQNAgent.load(checkpoint)
    if agent.config.to_dict() != ref["config"] or agent.config.to_dict() != ref5b["config"]:
        raise ValueError("Reference environment/configuration differs from checkpoint")
    seeds = ref["environment_seeds"]
    if seeds != ref5b["environment_seeds"] or seeds != list(range(agent.config.evaluation_seed_start,
            agent.config.evaluation_seed_start + agent.config.evaluation_episodes)):
        raise ValueError("Reference seeds differ from reserved evaluation range")
    if set(seeds) & set(agent.checkpoint_metadata.get("training_seeds", [])):
        raise ValueError("Evaluation seeds overlap training")
    weights = {key: tensor.clone() for key, tensor in agent.online.state_dict().items()}
    target_weights = {key: tensor.clone() for key, tensor in agent.target.state_dict().items()}
    progress = (agent.environment_steps, agent.optimiser_steps, agent._rng.getstate(), agent.epsilon)
    output.mkdir(parents=True)
    (output / "INCOMPLETE").write_text("Analysis is unverified until summary.json and report.md are written.\n")
    histories, episodes = [], []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(agent.config.cpu_threads)
        with gzip.GzipFile(filename=str(output/"state_traces.jsonl.gz"), mode="wb", compresslevel=6, mtime=0) as compressed:
            import io
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as trace, \
                 (output/"trap_events.csv").open("x", newline="", encoding="utf-8") as event_file:
                writer = csv.DictWriter(event_file, fieldnames=EVENT_FIELDS)
                writer.writeheader()
                env = SnakeEnv(agent.config.environment, agent.config.rewards)
                for number, seed in enumerate(seeds, 1):
                    episode, history = analyse_episode(env, agent, number, seed, trace, writer,
                        low_space_cells=low_space_cells, substantial_gap_cells=substantial_gap_cells)
                    episodes.append(episode)
                    histories.append(history)
                    if number % 10 == 0:
                        print(f"Analysed {number}/{len(seeds)} episodes", flush=True)
    finally:
        torch.set_num_threads(old_threads)
    metrics = summarise(episodes)
    verify_reproduction(episodes, reference, metrics)
    with (reference_5b/"episodes.csv").open(newline="") as handle:
        prior = list(csv.DictReader(handle))
    if len(prior) != len(episodes):
        raise ValueError("5B episode count mismatch")
    for actual, expected in zip(episodes, prior):
        for key in ("episode", "environment_seed", "score", "steps", "terminated", "truncated", "end_reason", "total_reward"):
            if str(actual[key]) != expected[key]:
                raise ValueError(f"5B episode reproduction mismatch: {key}")
    if metrics != ref5b["evaluation"]:
        raise ValueError("5B metrics differ")
    if any(not torch.equal(tensor, agent.online.state_dict()[key]) for key, tensor in weights.items()) or \
       any(not torch.equal(tensor, agent.target.state_dict()[key]) for key, tensor in target_weights.items()) or \
       progress != (agent.environment_steps, agent.optimiser_steps, agent._rng.getstate(), agent.epsilon):
        raise RuntimeError("Analysis changed model weights or training state")
    if hashes != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}:
        raise RuntimeError("A protected checkpoint/reference changed")
    with (output/"episodes.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    analysis = aggregate(episodes, histories, low_space_cells, substantial_gap_cells)
    examples = _examples(episodes)
    # Distance/remaining-length statistics read back the compact trace in a stream.
    distance_sums = [0, 0, 0]
    by_episode = {example["episode"]: example for example in examples}
    with gzip.open(output/"state_traces.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for index, value in enumerate(row["obstacle_free_cells"]):
                distance_sums[index] += value
            example = by_episode.get(row["episode"])
            if example is not None and row["space_deficit"] is not None:
                previous = example.get("largest_deficit_decision")
                if previous is None or row["space_deficit"] > previous["space_deficit"]:
                    example["largest_deficit_decision"] = decision_example(row, example["steps"])
    summary = {"schema_version": 1, "checkpoint": str(checkpoint), "config": agent.config.to_dict(),
        "checkpoint_sha256": hashes[str(checkpoint)], "environment_seeds": seeds, "epsilon": 0.,
        "analysis_config": {"low_space_cells": low_space_cells, "substantial_gap_cells": substantial_gap_cells,
            "distance_units": "free cells before current static wall/body, tail included",
            "reachable_units": "connected cells including candidate head, resulting body static",
            "candidate_collision_value": None, "lookahead": "not implemented (optional)"},
        "evaluation": metrics, "reproduces_milestone4_and_5b": True, "analysis": analysis,
        "mean_obstacle_free_cells": {name: distance_sums[index]/analysis["movement_count"]
                                     for index, name in enumerate(ACTION_NAMES)},
        "examples": examples, "model_and_training_state_unchanged": True,
        "protected_file_hashes": hashes, "python_version": platform.python_version(),
        "torch_version": str(torch.__version__)}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    write_report(output, summary)
    if make_charts:
        create_charts(output, examples, histories, low_space_cells)
    (output/"INCOMPLETE").unlink()
    print(json.dumps({"evaluation": metrics, "analysis": analysis, "mean_obstacle_free_cells":
                      summary["mean_obstacle_free_cells"]}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-5b", type=Path, default=DEFAULT_5B)
    args = parser.parse_args()
    run_analysis(args.checkpoint, args.output, args.reference, args.reference_5b)


if __name__ == "__main__":
    main()
