"""Static Matplotlib charts; episode and optimiser-step axes stay distinct."""

import argparse
import csv
from pathlib import Path
import statistics


def create_charts(run_dir: Path, rolling_window: int = 25) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    with (run_dir / "updates.csv").open(newline="", encoding="utf-8") as handle:
        updates = list(csv.DictReader(handle))
    complete = [row for row in episodes if row["episode_complete"] == "True"]
    numbers = [int(row["episode"]) for row in complete]
    scores = [float(row["score"]) for row in complete]
    rolling = [statistics.mean(scores[max(0, index - rolling_window + 1):index + 1])
               for index in range(len(scores))]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("SnakeAI DQN training — score is food collected, loss is prediction error")
    axes[0, 0].plot(numbers, scores, linewidth=0.8, color="#4169a1")
    axes[0, 0].set(title="Score per completed episode", xlabel="Episode", ylabel="Food collected")
    axes[0, 1].plot(numbers, rolling, color="#21855b")
    axes[0, 1].set(title=f"Rolling mean score (up to {rolling_window} episodes)", xlabel="Episode", ylabel="Mean food collected")
    axes[1, 0].plot(numbers, [float(row["episode_reward"]) for row in complete], linewidth=0.8)
    axes[1, 0].set(title="Reward per completed episode", xlabel="Episode", ylabel="Total reward")
    axes[1, 1].plot([int(row["optimiser_steps"]) for row in updates],
                    [float(row["loss"]) for row in updates], linewidth=0.5, alpha=0.8)
    axes[1, 1].set(title="Smooth L1 training loss", xlabel="Optimiser update", ylabel="Batch loss")
    axes[2, 0].plot([int(row["environment_steps"]) for row in episodes],
                    [float(row["epsilon"]) for row in episodes], color="#a35f21")
    axes[2, 0].set(title="Exploration at episode boundaries", xlabel="Environment step", ylabel="Epsilon", ylim=(0, 1.05))
    axes[2, 1].axis("off")
    axes[2, 1].text(0, 0.9, "Partial episodes excluded from score/reward plots.\n"
                            "Training loss alone does not demonstrate learning.\n"
                            "Assess the separate held-out greedy evaluation.", va="top")
    for axis in axes.flat:
        if axis.axison:
            axis.grid(alpha=0.2)
    destination = run_dir / "training_charts.png"
    if destination.exists():
        plt.close(fig)
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return [str(destination)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(create_charts(args.run_dir))


if __name__ == "__main__":
    main()
