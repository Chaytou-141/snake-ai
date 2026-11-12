"""One supervised regression step on invented examples, not DQN training."""

import json

import torch
from torch import nn

from snake_ai.models import DQNNetwork


def run_demo() -> dict[str, object]:
    torch.manual_seed(7)
    model = DQNNetwork().to(device="cpu")
    model.train()
    # Deterministic example features in the environment's usual 11-value order.
    observations = torch.tensor([
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    ], dtype=torch.float32, device="cpu")
    # These arbitrary teaching targets are NOT rewards or Bellman/DQN targets.
    targets = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 0.0],
                            [-1.0, 1.0, 0.5]], dtype=torch.float32, device="cpu")
    loss_function = nn.MSELoss()
    learning_rate = 0.001
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
    predictions = model(observations)             # Forward propagation.
    loss = loss_function(predictions, targets)    # Mean squared prediction error.
    loss_before = float(loss.detach().item())
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimiser.zero_grad()                        # Remove accumulated gradients.
    loss.backward()                              # Calculate gradients.
    gradient_norm = sum(float(parameter.grad.square().sum().item())
                        for parameter in model.parameters()
                        if parameter.grad is not None) ** 0.5
    optimiser.step()                             # Exactly one parameter update.
    with torch.no_grad():
        loss_after = float(loss_function(model(observations), targets).item())
    changed = sum(not torch.equal(old, parameter.detach())
                  for old, parameter in zip(before, model.parameters()))
    return {
        "description": "Educational regression only: invented targets, no RL or DQN training",
        "pytorch_version": torch.__version__, "device": "cpu", "model_seed": 7,
        "batch_shape": list(observations.shape), "targets": targets.tolist(),
        "optimiser": "SGD", "learning_rate": learning_rate, "optimiser_steps": 1,
        "loss": "mean squared error over 9 output values",
        "loss_before": loss_before, "loss_after": loss_after,
        "gradient_l2_norm": gradient_norm, "changed_parameter_tensors": changed,
        "note": "One optimiser step does not guarantee a lower loss for every configuration.",
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
