"""Standard DQN: online action regression against a detached target network."""

import os
from pathlib import Path
import random
import tempfile
from typing import Iterable

import torch
from torch import nn

from snake_ai.models import DQNNetwork
from snake_ai.training.config import TrainingConfig
from snake_ai.training.replay_buffer import ReplayBatch, ReplayBuffer, observation_values


def selected_q_values(all_q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """One predicted value per transition: Q(state, the action actually taken)."""
    return all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


class DQNAgent:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config if config is not None else TrainingConfig()
        # Constructor determinism does not alter the caller's torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.online = DQNNetwork(hidden_size=self.config.hidden_size).cpu()
            self.target = DQNNetwork(hidden_size=self.config.hidden_size).cpu()
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed + 1)
        self._rng = random.Random(self.config.seed + 2)
        self.environment_steps = 0
        self.optimiser_steps = 0
        self.episodes_completed = 0
        self.episodes_started = 0

    @property
    def epsilon(self) -> float:
        progress = min(self.environment_steps / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_min - self.config.epsilon_start)

    def predict(self, state: Iterable[float]) -> torch.Tensor:
        inputs = torch.tensor(observation_values(state), dtype=torch.float32)
        with torch.no_grad():
            return self.online(inputs).detach().clone()

    def act(self, state: Iterable[float], *, evaluate: bool = False) -> int:
        values = observation_values(state)
        # Greedy evaluation does not draw from or change training RNG/schedule.
        if not evaluate and self._rng.random() < self.epsilon:
            return self._rng.randrange(3)
        return int(self.predict(values).argmax().item())

    def remember(self, state, action, reward, next_state, terminated, truncated) -> None:
        self.replay.add(state, action, reward, next_state, terminated, truncated)
        self.environment_steps += 1

    def compute_targets(self, batch: ReplayBatch) -> torch.Tensor:
        with torch.no_grad():
            next_values = self.target(batch.next_states).max(dim=1).values
            # Truncation is a budget limit, not a genuine terminal state.
            return batch.rewards + self.config.gamma * (~batch.terminated).float() * next_values

    def train_step(self) -> dict[str, float | int] | None:
        if self.environment_steps < self.config.learning_starts or len(self.replay) < self.config.batch_size:
            return None
        self.online.train()
        batch = self.replay.sample(self.config.batch_size)
        predicted = selected_q_values(self.online(batch.states), batch.actions)
        targets = self.compute_targets(batch)
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; stopping experiment")
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm,
                                       error_if_nonfinite=True)
        self.optimiser.step()
        self.optimiser_steps += 1
        if self.optimiser_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach().item()), "gradient_norm_before_clip": float(norm.item()),
                "environment_steps": self.environment_steps, "optimiser_steps": self.optimiser_steps,
                "epsilon": self.epsilon}

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        """Atomic new checkpoint. Replay and active game are intentionally omitted."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "config": self.config.to_dict(),
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimiser": self.optimiser.state_dict(), "environment_steps": self.environment_steps,
            "optimiser_steps": self.optimiser_steps, "episodes_completed": self.episodes_completed,
            "episodes_started": self.episodes_started, "action_rng_state": self._rng.getstate(),
            "torch_rng_state": torch.get_rng_state(), "torch_version": str(torch.__version__),
            "resume_mode": "approximate: replay and active episode are not saved",
            "metadata": metadata if metadata is not None else {}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
            # Only a new path is used; previous verified checkpoints stay intact.
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, *, resume_training: bool = False) -> "DQNAgent":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        agent = cls(TrainingConfig.from_dict(payload["config"]))
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.environment_steps = payload["environment_steps"]
        agent.optimiser_steps = payload["optimiser_steps"]
        agent.episodes_completed = payload["episodes_completed"]
        agent.episodes_started = payload["episodes_started"]
        agent.checkpoint_metadata = payload["metadata"]
        if resume_training:
            agent.optimiser.load_state_dict(payload["optimiser"])
            agent._rng.setstate(payload["action_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"])
        else:
            agent.online.eval()
            agent.online.requires_grad_(False)
        return agent
