"""One headless environment action chosen by an UNTRAINED network."""

import json

import torch

from snake_ai.models import DQNNetwork
from snake_ai.rl import SnakeEnv, relative_direction


def run_demo(environment_seed: int = 42, model_seed: int = 7) -> dict[str, object]:
    env = SnakeEnv()
    observation, _ = env.reset(seed=environment_seed)
    torch.manual_seed(model_seed)
    model = DQNNetwork().to(device="cpu")
    model.eval()
    inputs = torch.tensor(list(observation), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        q_values = model(inputs)
    action = int(q_values.argmax(dim=-1).item())
    direction = relative_direction(env.game.direction, action)
    next_observation, reward, terminated, truncated, info = env.step(action)
    counts = model.parameter_counts()
    return {
        "description": "UNTRAINED network demonstration: random initial weights, no learned policy",
        "pytorch_version": torch.__version__, "device": "cpu",
        "environment_seed": environment_seed, "model_seed": model_seed,
        "observation": list(observation), "q_values": q_values.tolist(),
        "selected_action": action, "action_name": ("straight", "right", "left")[action],
        "absolute_direction": direction.name, "reward": reward,
        "terminated": terminated, "truncated": truncated,
        "next_observation": list(next_observation), "info": info,
        "parameters_per_layer": counts, "total_trainable_parameters": sum(counts.values()),
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
