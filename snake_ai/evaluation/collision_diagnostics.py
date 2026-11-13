"""Record final moves and classify immediate collision options without training."""

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import FEATURE_NAMES, SnakeEnv, relative_direction
from snake_ai.training.evaluate import summarise

ACTION_NAMES = ("straight", "right", "left")
CATEGORY_NAMES = {"A": "Avoidable immediate collision", "B": "No immediately safe action",
                  "C": "Other or uncertain"}
DEFAULT_CHECKPOINT = Path("results/milestone4_initial/checkpoints/final.pt")
DEFAULT_REFERENCE = Path("results/milestone4_evaluation")
DEFAULT_OUTPUT = Path("results/milestone5b_collision_diagnostics")
RETAINED_MOVES = 30
JSON_COLUMNS = {"head_before", "food_before", "body_before", "observation", "safe_actions",
                "attempted_head", "head_after", "food_after", "body_after"}


def snapshot(env: SnakeEnv) -> dict:
    """Copy actual state, with head-first body coordinates suitable for JSON."""
    game = env.game
    return {"head": list(game.snake[0]), "direction": game.direction.name,
            "food": list(game.food) if game.food is not None else None,
            "snake_length": len(game.snake), "score": game.score,
            "body": [list(position) for position in game.snake]}


def record_move(env: SnakeEnv, agent: DQNAgent, episode: int, seed: int) -> dict:
    """One actual greedy move, with three nonmutating pre-action safety checks."""
    before = snapshot(env)
    observation = list(env.observation())
    values = agent.predict(observation)
    if values.shape != (3,) or not torch.isfinite(values).all():
        raise ValueError("Diagnostic inference requires three finite Q-values")
    action = int(values.argmax().item())
    q_values = values.tolist()
    directions = [relative_direction(env.game.direction, index) for index in range(3)]
    danger = [env.game.would_collide(direction) for direction in directions]
    safe = [index for index in range(3) if not danger[index]]
    best_safe = max(safe, key=q_values.__getitem__) if safe else None
    dx, dy = directions[action].value
    attempted_head = [before["head"][0] + dx, before["head"][1] + dy]
    _, reward, terminated, truncated, info = env.step(action)
    after = snapshot(env)
    row = {"episode": episode, "environment_seed": seed, "move": info["steps"],
        **{f"{key}_before": value for key, value in before.items()},
        "observation": observation,
        **{f"q_{name}": q_values[index] for index, name in enumerate(ACTION_NAMES)},
        **{f"danger_{name}": danger[index] for index, name in enumerate(ACTION_NAMES)},
        "selected_action": action, "action_name": ACTION_NAMES[action],
        "selected_q": q_values[action], "selected_dangerous": danger[action],
        "safe_actions": safe, "best_safe_action": best_safe,
        "best_safe_q": q_values[best_safe] if best_safe is not None else None,
        "q_gap_selected_minus_best_safe": q_values[action] - q_values[best_safe] if best_safe is not None else None,
        "movement_direction": after["direction"], "attempted_head": attempted_head,
        "reward": reward, **{f"{key}_after": value for key, value in after.items()},
        "terminated": terminated, "truncated": truncated, "end_reason": info["end_reason"],
        "game_status_after": info["game_status"],
        "safety_matches_observation": observation[:3] == [float(value) for value in danger],
        "action_mapping_matches_engine": after["direction"] == directions[action].name}
    return row


def classify_end(row: dict) -> tuple[str, str]:
    """Classify only the final event, never retrospectively label earlier moves."""
    if not (row["terminated"] and not row["truncated"] and row["end_reason"] == "collision"
            and row["game_status_after"] == "lost"):
        return "C", "Episode ended without a genuine collision"
    if not (row["safety_matches_observation"] and row["action_mapping_matches_engine"]
            and row["selected_dangerous"]):
        return "C", "Collision evidence disagrees with predictor, observation, or action mapping"
    if row["safe_actions"]:
        return "A", "Selected collision while at least one other action was immediately safe"
    return "B", "All three relative actions predicted an immediate collision"


def diagnose_episode(env: SnakeEnv, agent: DQNAgent, episode: int, seed: int) -> tuple[dict, list[dict]]:
    env.reset(seed=seed)
    retained: deque[dict] = deque(maxlen=RETAINED_MOVES)
    total_reward = 0.0
    while True:
        row = record_move(env, agent, episode, seed)
        retained.append(row)
        total_reward += row["reward"]
        if row["terminated"] or row["truncated"]:
            break
    category, note = classify_end(row)
    episode_row = {"episode": episode, "environment_seed": seed, "score": row["score_after"],
        "steps": row["move"], "total_reward": total_reward, "terminated": row["terminated"],
        "truncated": row["truncated"], "end_reason": row["end_reason"],
        "collision_category": category, "classification_note": note,
        "retained_moves": len(retained)}
    return episode_row, list(retained)


def verify_reproduction(episodes: list[dict], reference: Path, metrics: dict) -> None:
    """Match every saved DQN episode, not only its aggregate mean."""
    with (reference / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        saved = [row for row in csv.DictReader(handle) if row["policy"] == "dqn_greedy"]
    if len(saved) != len(episodes):
        raise ValueError("Reproduction failed: reference episode count differs")
    for actual, expected in zip(episodes, saved):
        for field in ("episode", "environment_seed", "score", "steps"):
            if actual[field] != int(expected[field]):
                raise ValueError(f"Reproduction failed at seed {actual['environment_seed']}: {field}")
        for field in ("terminated", "truncated"):
            if str(actual[field]) != expected[field]:
                raise ValueError(f"Reproduction failed at seed {actual['environment_seed']}: {field}")
        if (actual["total_reward"] != float(expected["total_reward"])
                or actual["end_reason"] != expected["end_reason"]):
            raise ValueError(f"Reproduction failed at seed {actual['environment_seed']}: reward/end reason")
    recorded = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if metrics != recorded["summaries"]["dqn_greedy"]:
        raise ValueError("Reproduction failed: summary metrics differ")


def _stats(values: list[float]) -> dict:
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def _write_csv(path: Path, rows: list[dict], *, fields: list[str] | None = None) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if key in JSON_COLUMNS else value
                             for key, value in row.items()})


def _failure_examples(episodes: list[dict], tails: list[list[dict]]) -> dict:
    examples = {}
    for name, category in (("avoidable", "A"), ("no_safe_action", "B")):
        candidates = [index for index, row in enumerate(episodes) if row["collision_category"] == category
                      and row["end_reason"] == "collision"]
        examples[name] = candidates[0] if candidates else None
    collisions = [index for index, row in enumerate(episodes) if row["end_reason"] == "collision"]
    examples["highest_score"] = max(collisions, key=lambda index: episodes[index]["score"]) if collisions else None
    return {name: {"episode": episodes[index], "final_move": tails[index][-1],
                   "retained_moves": tails[index]} if index is not None else None
            for name, index in examples.items()}


def _write_report(output: Path, summary: dict, examples: dict) -> None:
    metrics = summary["evaluation"]
    lines = ["# SnakeAI — Milestone 5B collision diagnostics", "", "## Evaluation overview", "",
        f"Greedy inference on {metrics['episodes']} recorded evaluation seeds. Every episode's score, moves, "
        "reward, end flags, and end reason matched the original Milestone 4 CSV; all summary metrics matched.", "",
        f"Mean score **{metrics['average_score']}**, median **{metrics['median_score']}**, "
        f"maximum **{metrics['maximum_score']}**, mean moves **{metrics['average_episode_length']}**. "
        f"Collision rate **{metrics['collision_rate']:.0%}**. No training or policy changes.", "",
        "## Collision classification", "", "| Category | Count | Percentage of collisions |",
        "| --- | ---: | ---: |"]
    for category, label in CATEGORY_NAMES.items():
        item = summary["collision_categories"][category]
        lines.append(f"| {category} — {label} | {item['count']} | {item['percentage']:.1f}% |")
    lines += ["", "Type A means a safe *next move* existed; it does not show that taking it would avoid a later trap. "
              "Type B means no safe next move existed at the final state; it does not prove earlier decisions "
              "could not have prevented reaching that state. Type C covers inconsistent evidence or other endings. "
              "Win/truncation endings are reported separately from collision percentages.", "", "## Q-value analysis", ""]
    stats = summary["q_value_analysis"]["selected_minus_best_safe_for_type_a"]
    lines.append(f"The selected action was immediately dangerous in **{summary['q_value_analysis']['selected_dangerous_collisions']}** "
                 "collisions. All selections used the largest raw Q-value (ties choose the first action).")
    selected = summary["q_value_analysis"]["selected_q_at_collision"]
    if selected["count"]:
        lines.append(f"Selected collision Q-values averaged **{selected['mean']:.6f}**, range "
                     f"**{selected['minimum']:.6f} to {selected['maximum']:.6f}**. "
                     f"{summary['q_value_analysis']['nonnegative_selected_q_collisions']} were nonnegative. "
                     "These are predictions to inspect, not a probability or an independent safety classifier.")
    if stats["count"]:
        lines.append(f"For the {stats['count']} Type A collisions, selected Q minus the best safe alternative Q "
            f"averaged **{stats['mean']:.6f}**, median **{stats['median']:.6f}**, range "
            f"**{stats['minimum']:.6f} to {stats['maximum']:.6f}**.")
    else:
        lines.append("No Type A collisions were observed, so there is no selected-versus-safe Q gap to estimate.")
    lines += ["", "Q-values estimate discounted future reward, not probabilities or immediate collision risk. "
              "A negative value is not an independent safety test; safety here comes only from the engine predictor.",
              "", "## Failure examples", ""]
    for name, title in (("avoidable", "Safe alternative available"), ("no_safe_action", "No safe next action"),
                        ("highest_score", "Highest score before collision")):
        example = examples[name]
        lines.append(f"### {title}")
        lines.append("")
        if example is None:
            lines += ["No matching event occurred in this sample; an example cannot be supplied.", ""]
            continue
        episode, move = example["episode"], example["final_move"]
        qs = [move[f"q_{action}"] for action in ACTION_NAMES]
        safe = [ACTION_NAMES[action] for action in move["safe_actions"]]
        lines += [f"Episode **{episode['episode']}**, seed **{episode['environment_seed']}**, score "
            f"**{episode['score']}**, move **{move['move']}**, category **{episode['collision_category']}**. "
            f"Head {move['head_before']}, facing {move['direction_before']}; body length {move['snake_length_before']}.", "",
            "| Action | Q-value | Immediate collision |", "| --- | ---: | --- |"]
        for index, action in enumerate(ACTION_NAMES):
            selected = " (selected)" if index == move["selected_action"] else ""
            lines.append(f"| {action}{selected} | {qs[index]:.6f} | {move[f'danger_{action}']} |")
        lines += ["", f"Immediately safe alternatives: **{', '.join(safe) if safe else 'none'}**. "
                  f"Attempted destination {move['attempted_head']}; actual post-action head {move['head_after']}."]
        if move["best_safe_q"] is not None:
            lines.append(f"Selected Q {move['selected_q']:.6f}; best safe Q {move['best_safe_q']:.6f}; "
                         f"gap {move['q_gap_selected_minus_best_safe']:.6f}.")
        lines += ["This supports the immediate option comparison only. It does not identify the cause of "
                  "earlier decisions, prove a safe alternative would survive, or establish that missing board "
                  "information caused this collision.", ""]
    lines += ["Examples select the first observed A and B events and the maximum-score collision (first on ties); "
              "roles may overlap. Full body coordinates and final 30 moves are in `example_failure.json`.", "",
        "## Schema and reading the files", "",
        "- `episodes.csv`: episode, seed, final score, steps, total reward, end flags/reason, category/note, retained count.",
        "- `final_moves.csv`: at most 30 actual moves per episode. `*_before`/`*_after` fields give "
        "head, direction, food, length, score, and complete head-first body. `observation` follows the existing "
        "11-feature order recorded in summary.json. `q_straight/right/left`, `selected_action` (0/1/2), "
        "`action_name`, and actual `movement_direction` describe the decision. `danger_*` are pre-action "
        "engine predictions. `safe_actions`, best-safe values, Q gap, reward, end flags/reason, and consistency "
        "flags describe the outcome. `attempted_head` is a projected destination, not the actual post-state.",
        "- `collisions.csv`: the final move of each collision with category/note. Same move schema as above.",
        "- Coordinates, bodies, observations, and safe-action lists are JSON encoded inside CSV cells; absent "
        "food is JSON `null`. CSV booleans are `True`/`False`, nullable scalar fields are blank, and move/episode "
        "numbers are one-based. Coordinates are (column, row), origin top-left. A collision leaves the actual "
        "body/head in its last valid position, per the existing engine.",
        "- `summary.json`: configuration, hashes, exact seeds, reproduction status, statistics and preservation checks.",
        "- `example_failure.json`: three named example roles, each containing an episode row, final move, "
        "and retained moves; a missing category has JSON `null`.", "",
        "## Limitations and next experiments", "",
        "This is one already-trained model and 100 previously evaluated seeds, not fresh generalisation evidence. "
        "The 11 inputs omit the full board/body geometry, distances, and budgets. Immediate safety cannot reveal "
        "long-term trapping; imperfect Q estimates can also rank actions poorly. These are possible mechanisms, "
        "not demonstrated causes for individual failures. Only the last 30 moves are retained, so earlier setup "
        "may be missing. The extra diagnostic body data was never supplied to the policy.", "",
        "Next, inspect earlier trap formation with explicitly defined geometry/planning measurements, then "
        "pre-register any observation or algorithm comparison and evaluate multiple independent training seeds "
        "on fresh untouched seeds. Type A failures motivate value-ranking/calibration checks; Type B failures "
        "motivate investigation of earlier planning. This run implements no improvements or training experiments.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _chart(output: Path, categories: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    counts = [categories[category]["count"] for category in CATEGORY_NAMES]
    bars = ax.bar(["A: safe alternative", "B: no safe next move", "C: uncertain"], counts,
                  color=["#d48944", "#477da8", "#8a8c92"])
    ax.bar_label(bars)
    ax.set(title="Final collision options — immediate safety only", ylabel="Collision count",
           ylim=(0, max(counts, default=0) * 1.15 + 1))
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.savefig(output / "collision_categories.png", dpi=150)
    plt.close(fig)


def run_diagnostics(checkpoint: Path = DEFAULT_CHECKPOINT, output: Path = DEFAULT_OUTPUT,
                    reference: Path = DEFAULT_REFERENCE, *, make_chart: bool = True) -> dict:
    checkpoint, output, reference = Path(checkpoint), Path(output), Path(reference)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {output}")
    protected = [checkpoint, reference / "episodes.csv", reference / "summary.json"]
    before_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    recorded = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if before_files[str(checkpoint)] != recorded["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from the reference evaluation")
    agent = DQNAgent.load(checkpoint)
    if agent.config.to_dict() != recorded["config"]:
        raise ValueError("Checkpoint configuration differs from the reference evaluation")
    seeds = recorded["environment_seeds"]
    expected_seeds = list(range(agent.config.evaluation_seed_start,
                                agent.config.evaluation_seed_start + agent.config.evaluation_episodes))
    if seeds != expected_seeds or set(seeds) & set(agent.checkpoint_metadata.get("training_seeds", [])):
        raise ValueError("Reference seeds differ from the reserved held-out range or overlap training")
    weights = {name: tensor.clone() for name, tensor in agent.online.state_dict().items()}
    targets = {name: tensor.clone() for name, tensor in agent.target.state_dict().items()}
    progress = (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())
    env = SnakeEnv(agent.config.environment, agent.config.rewards)
    old_threads = torch.get_num_threads()
    episodes, tails = [], []
    try:
        torch.set_num_threads(agent.config.cpu_threads)
        for number, seed in enumerate(seeds, 1):
            episode, retained = diagnose_episode(env, agent, number, seed)
            episodes.append(episode)
            tails.append(retained)
    finally:
        torch.set_num_threads(old_threads)
    metrics = summarise(episodes)
    verify_reproduction(episodes, reference, metrics)  # No analysis artifacts before this passes.
    if (any(not torch.equal(value, agent.online.state_dict()[name]) for name, value in weights.items())
            or any(not torch.equal(value, agent.target.state_dict()[name]) for name, value in targets.items())
            or progress != (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())):
        raise RuntimeError("Diagnostics changed model weights or training state")
    after_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    if before_files != after_files:
        raise RuntimeError("Checkpoint or reference files changed")
    collisions = [{**moves[-1], "collision_category": episode["collision_category"],
                   "classification_note": episode["classification_note"]}
                  for episode, moves in zip(episodes, tails) if episode["end_reason"] == "collision"]
    counts = Counter(row["collision_category"] for row in collisions)
    categories = {category: {"label": label, "count": counts[category],
        "percentage": 100 * counts[category] / len(collisions) if collisions else 0.0}
        for category, label in CATEGORY_NAMES.items()}
    gaps = [row["q_gap_selected_minus_best_safe"] for row in collisions if row["collision_category"] == "A"]
    summary = {"schema_version": 1, "checkpoint": str(checkpoint),
        "checkpoint_sha256": before_files[str(checkpoint)], "reference": str(reference),
        "config": agent.config.to_dict(), "environment_seeds": seeds, "epsilon": 0.0,
        "feature_order": list(FEATURE_NAMES), "action_order": list(ACTION_NAMES),
        "retained_moves_per_episode_limit": RETAINED_MOVES,
        "retained_move_count": sum(map(len, tails)), "evaluation": metrics,
        "reproduction_verified": True, "reproduction_checked": "all per-episode outcomes and aggregate metrics",
        "collision_count": len(collisions), "collision_categories": categories,
        "episode_end_categories": dict(Counter(row["collision_category"] for row in episodes)),
        "q_value_analysis": {"selected_dangerous_collisions": sum(row["selected_dangerous"] for row in collisions),
            "selected_q_at_collision": _stats([row["selected_q"] for row in collisions]),
            "nonnegative_selected_q_collisions": sum(row["selected_q"] >= 0 for row in collisions),
            "selected_minus_best_safe_for_type_a": _stats(gaps)},
        "model_weights_unchanged": True, "training_state_unchanged": True,
        "reference_and_checkpoint_files_unchanged": True, "protected_file_hashes": before_files,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__)}
    examples = _failure_examples(episodes, tails)
    output.mkdir(parents=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "final_moves.csv", [row for moves in tails for row in moves])
    collision_fields = list(tails[0][-1]) + ["collision_category", "classification_note"]
    _write_csv(output / "collisions.csv", collisions, fields=collision_fields)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "example_failure.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    _write_report(output, summary, examples)
    if make_chart:
        _chart(output, categories)
    print(json.dumps({"evaluation": metrics, "collision_categories": categories,
                      "q_value_analysis": summary["q_value_analysis"]}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    run_diagnostics(args.checkpoint, args.output, args.reference)


if __name__ == "__main__":
    main()
