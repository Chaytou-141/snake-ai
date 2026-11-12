"""Greedy held-out DQN evaluation paired with a fresh random policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import RandomAgent, SnakeEnv


def summarise(rows: list[dict]) -> dict:
    scores = [row["score"] for row in rows]
    count = len(rows)
    return {"episodes": count, "average_score": statistics.mean(scores),
        "median_score": statistics.median(scores), "minimum_score": min(scores),
        "maximum_score": max(scores),
        "average_episode_length": statistics.mean(row["steps"] for row in rows),
        "collision_rate": sum(row["end_reason"] == "collision" for row in rows) / count,
        "truncation_rate": sum(row["truncated"] for row in rows) / count,
        "win_rate": sum(row["end_reason"] == "win" for row in rows) / count}


def evaluate_agent(agent: DQNAgent) -> tuple[list[dict], dict]:
    config = agent.config
    seeds = list(range(config.evaluation_seed_start,
                       config.evaluation_seed_start + config.evaluation_episodes))
    used_seeds = set(getattr(agent, "checkpoint_metadata", {}).get("training_seeds", []))
    if used_seeds.intersection(seeds):
        raise ValueError("Evaluation seeds overlap recorded training seeds")
    before_weights = {key: value.clone() for key, value in agent.online.state_dict().items()}
    before_target = {key: value.clone() for key, value in agent.target.state_dict().items()}
    before_progress = (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())
    rows = []
    for policy in ("dqn_greedy", "random"):
        env = SnakeEnv(config.environment, config.rewards)
        random_agent = RandomAgent()
        for index, seed in enumerate(seeds):
            observation, _ = env.reset(seed=seed)
            random_seed = config.random_policy_seed_start + index
            random_agent.reset(seed=random_seed)
            total_reward = 0.0
            while True:
                action = agent.act(observation, evaluate=True) if policy == "dqn_greedy" else random_agent.act()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rows.append({"policy": policy, "episode": index + 1, "environment_seed": seed,
                "random_policy_seed": random_seed if policy == "random" else "",
                "score": info["score"], "steps": info["steps"], "total_reward": total_reward,
                "terminated": terminated, "truncated": truncated, "end_reason": info["end_reason"]})
    if (not all(torch.equal(before_weights[key], value) for key, value in agent.online.state_dict().items())
            or not all(torch.equal(before_target[key], value) for key, value in agent.target.state_dict().items())
            or before_progress != (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())):
        raise RuntimeError("Evaluation changed training state")
    summaries = {policy: summarise([row for row in rows if row["policy"] == policy])
                 for policy in ("dqn_greedy", "random")}
    difference = {key: summaries["dqn_greedy"][key] - summaries["random"][key]
                  for key in summaries["random"] if key != "episodes"}
    manifest = {"environment_seeds": seeds, "random_policy_seeds": list(range(
            config.random_policy_seed_start, config.random_policy_seed_start + len(seeds))),
        "config": config.to_dict(), "dqn_epsilon": 0.0, "weights_unchanged": True,
        "summaries": summaries, "difference_dqn_minus_random": difference,
        "interpretation": "One training seed, initial bounded experiment; not evidence of robust generalisation.",
        "earlier_baseline": "results/random_baseline.csv used different seeds and is not this paired comparison"}
    return rows, manifest


def run_evaluation(checkpoint: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {output}")
    agent = DQNAgent.load(checkpoint)
    torch.set_num_threads(agent.config.cpu_threads)
    rows, manifest = evaluate_agent(agent)
    manifest["checkpoint"] = str(checkpoint)
    manifest["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    manifest["torch_version"] = str(torch.__version__)
    output.mkdir(parents=True)
    with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": manifest["summaries"],
                      "difference_dqn_minus_random": manifest["difference_dqn_minus_random"]}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
