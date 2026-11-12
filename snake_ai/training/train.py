"""Bounded headless CPU training: python -m snake_ai.training.train --help."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.rl import SnakeEnv
from snake_ai.training.config import TrainingConfig

EPISODE_FIELDS = ["episode", "environment_seed", "score", "episode_reward", "episode_length",
                  "epsilon", "mean_training_loss", "end_reason", "terminated", "truncated",
                  "episode_complete", "environment_steps", "optimiser_steps"]
UPDATE_FIELDS = ["loss", "gradient_norm_before_clip", "environment_steps", "optimiser_steps", "epsilon"]


def run_training(config: TrainingConfig, output: Path, *, resume: Path | None = None,
                 make_charts: bool = True) -> dict:
    output = Path(output)
    # Every run owns a fresh directory; results and checkpoints are never replaced.
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    agent = DQNAgent.load(resume, resume_training=True) if resume is not None else DQNAgent(config)
    if agent.config != config:
        raise ValueError("Approximate resume requires the original checkpoint configuration")
    if config.training_seed_start + agent.episodes_started + config.episodes > config.evaluation_seed_start:
        raise ValueError("Run would enter the reserved evaluation seed range")
    output.mkdir(parents=True)
    torch.set_num_threads(config.cpu_threads)
    initial_weights = [parameter.detach().clone() for parameter in agent.online.parameters()]
    env = SnakeEnv(config.environment, config.rewards)
    prior_seeds = list(getattr(agent, "checkpoint_metadata", {}).get("training_seeds", []))
    training_seeds = prior_seeds.copy()
    rows = []
    status = "episode_budget"
    start = time.perf_counter()
    run_steps = 0
    initial_updates = agent.optimiser_steps
    episode_length = 0
    episode_reward = 0.0
    episode_losses: list[float] = []
    current_seed = None
    info = {"score": 0}
    manifest = {"config": config.to_dict(), "python_version": platform.python_version(),
                "torch_version": str(torch.__version__), "device": "cpu",
                "resume_from": str(resume) if resume is not None else None,
                "resume_mode": "approximate; replay and active game are not restored" if resume else "fresh",
                "evaluation_seeds": list(range(config.evaluation_seed_start,
                    config.evaluation_seed_start + config.evaluation_episodes))}
    (output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with (output / "episodes.csv").open("x", newline="", encoding="utf-8") as episode_file, \
         (output / "updates.csv").open("x", newline="", encoding="utf-8") as update_file:
        episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
        update_writer = csv.DictWriter(update_file, fieldnames=UPDATE_FIELDS)
        episode_writer.writeheader()
        update_writer.writeheader()

        def log_episode(reason, terminated, truncated, complete):
            row = {"episode": agent.episodes_started, "environment_seed": current_seed,
                "score": info["score"], "episode_reward": episode_reward,
                "episode_length": episode_length, "epsilon": agent.epsilon,
                "mean_training_loss": statistics.mean(episode_losses) if episode_losses else "",
                "end_reason": reason, "terminated": terminated, "truncated": truncated,
                "episode_complete": complete, "environment_steps": agent.environment_steps,
                "optimiser_steps": agent.optimiser_steps}
            rows.append(row)
            episode_writer.writerow(row)
            episode_file.flush()
            update_file.flush()

        try:
            for _ in range(config.episodes):
                current_seed = config.training_seed_start + agent.episodes_started
                agent.episodes_started += 1
                training_seeds.append(current_seed)
                observation, info = env.reset(seed=current_seed)
                episode_reward = 0.0
                episode_length = 0
                episode_losses = []
                while True:
                    action = agent.act(observation)
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    run_steps += 1
                    episode_length += 1
                    episode_reward += reward
                    cap = run_steps >= config.max_environment_steps
                    agent.remember(observation, action, reward, next_observation,
                                   terminated, truncated or (cap and not terminated))
                    observation = next_observation
                    update = agent.train_step()
                    if update is not None:
                        episode_losses.append(update["loss"])
                        update_writer.writerow(update)
                    if terminated or truncated or cap:
                        complete = terminated or truncated
                        if complete:
                            agent.episodes_completed += 1
                        log_episode(info["end_reason"] if complete else "run_step_cap",
                                    terminated, truncated or (cap and not terminated), complete)
                        episode_length = 0  # Prevent duplicate partial row if interrupted between episodes.
                        break
                if len(rows) % config.checkpoint_interval_episodes == 0:
                    agent.save(output / "checkpoints" / f"episode-{agent.episodes_started:06d}.pt",
                               metadata={"training_seeds": training_seeds})
                    print(f"Episodes {len(rows)} | steps {run_steps} | score {info['score']} | "
                          f"epsilon {agent.epsilon:.3f} | updates {agent.optimiser_steps}", flush=True)
                if cap:
                    status = "environment_step_cap"
                    break
        except (KeyboardInterrupt, FloatingPointError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and "non-finite" not in str(error).lower():
                # Preserve progress before reporting an unexpected runtime error too.
                status = f"runtime_error: {error}"
            else:
                status = "interrupted" if isinstance(error, KeyboardInterrupt) else f"unstable: {error}"
            if episode_length:
                log_episode(status, False, True, False)
            print(f"Stopped safely: {status}", flush=True)

    duration = time.perf_counter() - start
    complete_rows = [row for row in rows if row["episode_complete"]]
    scores = [row["score"] for row in complete_rows]
    summary = {"status": status, "duration_seconds": duration,
        "completed_episodes_this_run": len(complete_rows), "logged_episode_rows": len(rows),
        "environment_steps_this_run": run_steps, "environment_steps_total": agent.environment_steps,
        "optimiser_updates_this_run": agent.optimiser_steps - initial_updates,
        "optimiser_updates_total": agent.optimiser_steps, "replay_size": len(agent.replay),
        "final_epsilon": agent.epsilon,
        "score_mean": statistics.mean(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None,
        "last_100_mean_score": statistics.mean(scores[-100:]) if scores else None,
        "weights_changed": any(not torch.equal(old, parameter) for old, parameter in
                               zip(initial_weights, agent.online.parameters())),
        "training_seeds": training_seeds, "checkpoint": str(output / "checkpoints/final.pt")}
    agent.save(output / "checkpoints/final.pt", metadata={"training_seeds": training_seeds,
               "training_summary": summary})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if make_charts:
        from snake_ai.training.charts import create_charts
        create_charts(output)
    print(json.dumps({key: value for key, value in summary.items() if key != "training_seeds"}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dqn.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume-approx", type=Path)
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    overrides = {name: value for name, value in
        (("episodes", args.episodes), ("max_environment_steps", args.max_steps),
         ("learning_starts", args.learning_starts), ("batch_size", args.batch_size)) if value is not None}
    config = replace(config, **overrides)
    run_training(config, args.output, resume=args.resume_approx)


if __name__ == "__main__":
    main()
