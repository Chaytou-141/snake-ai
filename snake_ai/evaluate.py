"""Run with python -m snake_ai.evaluate; stores CSV episodes and JSON manifest."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import statistics

from snake_ai.rl import EnvConfig, RandomAgent, RewardConfig, SnakeEnv


def evaluate(episodes: int = 100, seed_start: int = 0,
             agent_seed_start: int = 10_000, config: EnvConfig | None = None,
             rewards: RewardConfig | None = None) -> tuple[list[dict], dict]:
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    env = SnakeEnv(config=config, rewards=rewards)
    agent = RandomAgent()
    rows = []
    for episode in range(episodes):
        env_seed = seed_start + episode
        agent_seed = agent_seed_start + episode
        env.reset(seed=env_seed)
        agent.reset(seed=agent_seed)
        total_reward = 0.0
        while True:
            _, reward, terminated, truncated, info = env.step(agent.act())
            total_reward += reward
            if terminated or truncated:
                break
        rows.append({"episode": episode + 1, "environment_seed": env_seed,
                     "agent_seed": agent_seed, "score": info["score"],
                     "steps": info["steps"], "total_reward": total_reward,
                     "terminated": terminated, "truncated": truncated,
                     "end_reason": info["end_reason"]})
    scores = [row["score"] for row in rows]
    summary = {"episodes": episodes, "average_score": statistics.mean(scores),
               "median_score": statistics.median(scores), "maximum_score": max(scores),
               "minimum_score": min(scores),
               "average_episode_length": statistics.mean(row["steps"] for row in rows),
               "collision_rate": sum(row["end_reason"] == "collision" for row in rows) / episodes,
               "truncation_rate": sum(row["truncated"] for row in rows) / episodes,
               "win_rate": sum(row["end_reason"] == "win" for row in rows) / episodes}
    manifest = {"format_version": 1, "python_version": platform.python_version(),
                "policy": "uniform random over actions 0, 1, 2; independently seeded per episode",
                "environment_config": asdict(env.config), "rewards": asdict(env.rewards),
                "environment_seeds": [row["environment_seed"] for row in rows],
                "agent_seeds": [row["agent_seed"] for row in rows],
                "summary": summary,
                "interpretation": "Initial baseline; 100 episodes are not a definitive performance estimate."}
    return rows, manifest


def save_results(output: Path, rows: list[dict], manifest: dict) -> None:
    """Do not overwrite existing experiments. Use another output name to rerun."""
    output = Path(output)
    metadata_path = output.with_suffix(".json")
    if output.suffix.lower() != ".csv":
        raise ValueError("output must have a .csv extension")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Results already exist; choose a new --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--agent-seed-start", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--starvation-min-steps", type=int, default=400)
    parser.add_argument("--starvation-factor", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/random_baseline.csv"))
    args = parser.parse_args()
    try:
        config = EnvConfig(max_steps=args.max_steps,
                           starvation_min_steps=args.starvation_min_steps,
                           starvation_factor=args.starvation_factor)
        rows, manifest = evaluate(args.episodes, args.seed_start, args.agent_seed_start, config)
        save_results(args.output, rows, manifest)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
