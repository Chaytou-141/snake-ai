"""An untrained Q-value network; no DQN learning algorithm is implemented."""

import torch
from torch import nn


class DQNNetwork(nn.Module):
    """Two ReLU hidden layers and unrestricted action-value outputs.

    Accept a floating-point tensor shaped (input_dim,) or (batch, input_dim).
    The caller controls conversion, seeds, dtype, and device. By default the
    module is created on CPU in float32, as used by both educational demos.
    """

    def __init__(self, input_dim: int = 11, hidden_size: int = 128,
                 output_dim: int = 3) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_size", hidden_size),
                            ("output_dim", output_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim not in (1, 2) or observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected shape ({self.input_dim},) or (batch, {self.input_dim}); "
                f"received {tuple(observation.shape)}"
            )
        if not observation.is_floating_point():
            raise TypeError("observation must be floating point; use dtype=torch.float32")
        hidden = self.relu(self.input_layer(observation))
        hidden = self.relu(self.hidden_layer(hidden))
        # Q-values are regression outputs, not probabilities: no Softmax.
        return self.output_layer(hidden)

    def parameter_counts(self) -> dict[str, int]:
        """Count actual trainable tensors, grouped by layer (no hardcoded total)."""
        return {
            name: sum(parameter.numel() for parameter in layer.parameters()
                      if parameter.requires_grad)
            for name, layer in self.named_children() if isinstance(layer, nn.Linear)
        }
