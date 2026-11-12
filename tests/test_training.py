"""DQN unit and small pipeline tests. No held-out experiment tuning."""

from dataclasses import replace
import json
import math
import random

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent, selected_q_values
from snake_ai.training.config import TrainingConfig
from snake_ai.training.evaluate import evaluate_agent
from snake_ai.training.replay_buffer import ReplayBuffer, ReplayBatch
from snake_ai.training.train import run_training

STATE = [0.0] * 11


@pytest.fixture(autouse=True)
def deterministic_cpu():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(original_threads)


def small_config(**kwargs):
    return replace(TrainingConfig(), batch_size=4, learning_starts=4,
                   replay_capacity=20, target_update_interval=2, **kwargs)


def fill(agent, count=8):
    for index in range(count):
        agent.remember(STATE, index % 3, 1.0, STATE, False, False)


def test_replay_is_bounded_copied_and_correctly_typed():
    replay = ReplayBuffer(capacity=2, seed=7)
    state = STATE.copy()
    for index in range(3):
        replay.add(state, index, index, state, index == 2, False)
    state[0] = 99
    batch = replay.sample(2)
    assert len(replay) == 2
    assert set(batch.actions.tolist()) == {1, 2}  # Oldest experience evicted.
    assert batch.states.shape == batch.next_states.shape == (2, 11)
    assert batch.states.dtype == batch.rewards.dtype == torch.float32
    assert batch.actions.dtype == torch.int64
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    assert not batch.states.any()
    assert batch.terminated.sum() == 1


def test_replay_seed_and_global_random_independence():
    global_state = random.getstate()
    first, second = ReplayBuffer(seed=7), ReplayBuffer(seed=7)
    for index in range(10):
        for replay in (first, second):
            replay.add(STATE, index % 3, index, STATE, False, False)
    for _ in range(3):
        assert torch.equal(first.sample(4).rewards, second.sample(4).rewards)
    assert random.getstate() == global_state


@pytest.mark.parametrize("size", [0, -1, 1, True])
def test_insufficient_replay_samples(size):
    with pytest.raises(ValueError):
        ReplayBuffer().sample(size)


@pytest.mark.parametrize("state, action, reward, terminated, truncated", [
    ([0] * 10, 0, 0, False, False), ([float("nan")] * 11, 0, 0, False, False),
    (STATE, True, 0, False, False), (STATE, 3, 0, False, False),
    (STATE, 0, float("inf"), False, False), (STATE, 0, 0, 1, False),
    (STATE, 0, 0, True, True),
])
def test_invalid_replay_transition(state, action, reward, terminated, truncated):
    replay = ReplayBuffer()
    with pytest.raises(ValueError):
        replay.add(state, action, reward, STATE, terminated, truncated)
    assert len(replay) == 0


def test_selected_actions_gather_not_max_or_broadcast():
    outputs = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], requires_grad=True)
    gathered = selected_q_values(outputs, torch.tensor([0, 1]))
    assert torch.equal(gathered, torch.tensor([1.0, 20.0]))
    gathered.sum().backward()
    assert torch.equal(outputs.grad, torch.tensor([[1., 0., 0.], [0., 1., 0.]]))


def test_terminal_and_truncated_targets_use_target_network_without_gradients():
    agent = DQNAgent(small_config())
    with torch.no_grad():
        for net in (agent.online, agent.target):
            for parameter in net.parameters():
                parameter.zero_()
        agent.target.output_layer.bias.copy_(torch.tensor([1., 5., 2.]))
        agent.online.output_layer.bias.fill_(100.)  # Must not be used for targets.
    batch = ReplayBatch(torch.zeros(3, 11), torch.zeros(3, dtype=torch.int64),
        torch.tensor([2., 2., -10.]), torch.zeros(3, 11),
        torch.tensor([False, False, True]), torch.tensor([False, True, False]))
    targets = agent.compute_targets(batch)
    assert torch.allclose(targets, torch.tensor([6.95, 6.95, -10.]))
    assert not targets.requires_grad
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in agent.target.parameters())


def test_epsilon_one_explores_seeded_uniform_actions():
    config = small_config(epsilon_start=1.0, epsilon_min=1.0)
    first, second = DQNAgent(config), DQNAgent(config)
    actions = [first.act(STATE) for _ in range(100)]
    assert actions == [second.act(STATE) for _ in range(100)]
    assert set(actions) == {0, 1, 2}
    assert first.environment_steps == 0


def test_greedy_evaluation_does_not_advance_schedule_or_rng():
    agent = DQNAgent(small_config())
    with torch.no_grad():
        agent.online.output_layer.weight.zero_()
        agent.online.output_layer.bias.copy_(torch.tensor([-1., 5., 2.]))
    before = (agent.epsilon, agent.environment_steps, agent._rng.getstate())
    assert all(agent.act(STATE, evaluate=True) == 1 for _ in range(10))
    assert before == (agent.epsilon, agent.environment_steps, agent._rng.getstate())
    agent = DQNAgent(small_config(epsilon_start=0.0, epsilon_min=0.0))
    assert agent.act(STATE) == int(agent.predict(STATE).argmax().item())


def test_epsilon_decay_counts_environment_steps_not_updates():
    agent = DQNAgent(small_config(epsilon_decay_steps=10))
    assert agent.epsilon == 1.0
    fill(agent, 5)
    assert agent.epsilon == pytest.approx(0.525)
    for _ in range(3):
        agent.train_step()
    assert agent.epsilon == pytest.approx(0.525)
    fill(agent, 20)
    assert agent.epsilon == pytest.approx(0.05)


def test_learning_warmup_updates_and_target_synchronisation():
    agent = DQNAgent(small_config(gradient_clip_norm=0.001))
    assert agent.train_step() is None
    fill(agent)
    original = [parameter.clone() for parameter in agent.target.parameters()]
    update = agent.train_step()
    assert math.isfinite(update["loss"])
    assert update["optimiser_steps"] == 1
    assert all(torch.equal(old, new) for old, new in zip(original, agent.target.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(original, agent.online.parameters()))
    clipped_norm = sum(parameter.grad.square().sum() for parameter in agent.online.parameters()).sqrt()
    assert clipped_norm <= agent.config.gradient_clip_norm + 1e-5
    assert update["gradient_norm_before_clip"] > agent.config.gradient_clip_norm
    agent.train_step()
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))
    assert all(parameter.grad is None for parameter in agent.target.parameters())


def test_seeded_updates_reproduce():
    first, second = DQNAgent(small_config()), DQNAgent(small_config())
    fill(first)
    fill(second)
    for _ in range(3):
        assert first.train_step() == second.train_step()
    assert all(torch.equal(a, b) for a, b in zip(first.online.parameters(), second.online.parameters()))


def test_checkpoint_inference_and_approximate_resume(tmp_path):
    agent = DQNAgent(small_config())
    fill(agent)
    agent.train_step()
    agent.act(STATE)
    path = tmp_path / "checkpoint.pt"
    agent.save(path, metadata={"training_seeds": [100000]})
    inference = DQNAgent.load(path)
    resumed = DQNAgent.load(path, resume_training=True)
    assert torch.equal(agent.predict(STATE), inference.predict(STATE))
    assert not any(parameter.requires_grad for parameter in inference.online.parameters())
    assert resumed.environment_steps == agent.environment_steps
    assert resumed.optimiser_steps == agent.optimiser_steps
    assert resumed.epsilon == agent.epsilon
    assert resumed._rng.getstate() == agent._rng.getstate()
    assert len(resumed.replay) == 0  # Explicitly approximate: no replay restoration.
    assert resumed.optimiser.state_dict()["state"]
    assert all(parameter.requires_grad for parameter in resumed.online.parameters())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        agent.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_short_training_pipeline_and_nonmutating_evaluation(tmp_path):
    config = small_config(episodes=6, max_environment_steps=60, evaluation_episodes=3)
    output = tmp_path / "run"
    summary = run_training(config, output, make_charts=False)
    assert summary["environment_steps_this_run"] <= 60
    assert summary["optimiser_updates_this_run"] > 0
    assert summary["weights_changed"]
    assert summary["replay_size"] > 0
    agent = DQNAgent.load(output / "checkpoints/final.pt")
    rows, manifest = evaluate_agent(agent)
    assert len(rows) == 6
    assert manifest["weights_unchanged"]
    assert manifest["environment_seeds"] == [900000, 900001, 900002]
    assert not set(summary["training_seeds"]) & set(manifest["environment_seeds"])
    assert json.loads((output / "summary.json").read_text())["weights_changed"]
    with pytest.raises(FileExistsError):
        run_training(config, output, make_charts=False)


def test_budget_cap_marks_partial_episode_and_transition_truncated(tmp_path):
    config = small_config(episodes=5, max_environment_steps=1)
    summary = run_training(config, tmp_path / "cap", make_charts=False)
    assert summary["environment_steps_this_run"] == 1
    assert summary["completed_episodes_this_run"] == 0
    assert summary["status"] == "environment_step_cap"
    import csv
    rows = list(csv.DictReader((tmp_path / "cap/episodes.csv").open()))
    assert rows[0]["truncated"] == "True"
    assert rows[0]["episode_complete"] == "False"


def test_interruption_preserves_checkpoint_and_partial_metrics(tmp_path, monkeypatch):
    original = DQNAgent.act
    calls = 0
    def interrupt(self, state, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(self, state, **kwargs)
    monkeypatch.setattr(DQNAgent, "act", interrupt)
    summary = run_training(small_config(), tmp_path / "interrupt", make_charts=False)
    assert summary["status"] == "interrupted"
    assert summary["environment_steps_this_run"] == 2
    loaded = DQNAgent.load(tmp_path / "interrupt/checkpoints/final.pt")
    assert loaded.environment_steps == 2


@pytest.mark.parametrize("kwargs", [{"gamma": 1.1}, {"learning_rate": 0},
    {"batch_size": 50001}, {"epsilon_min": 1.1}, {"cpu_threads": True},
    {"training_seed_start": 900000}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_central_config_roundtrip():
    from pathlib import Path
    config = TrainingConfig.load(Path(__file__).resolve().parents[1] / "configs/dqn.json")
    assert config == TrainingConfig()
    assert TrainingConfig.from_dict(config.to_dict()) == config


def test_training_pipeline_is_reproducible(tmp_path):
    config = small_config(episodes=2, max_environment_steps=20)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_training(config, first, make_charts=False)
    run_training(config, second, make_charts=False)
    assert (first / "episodes.csv").read_bytes() == (second / "episodes.csv").read_bytes()
    assert (first / "updates.csv").read_bytes() == (second / "updates.csv").read_bytes()
    a = DQNAgent.load(first / "checkpoints/final.pt")
    b = DQNAgent.load(second / "checkpoints/final.pt")
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))


def test_approximate_resume_has_empty_replay_and_uses_next_seed(tmp_path):
    config = small_config(episodes=2, max_environment_steps=10)
    first = run_training(config, tmp_path / "first", make_charts=False)
    second = run_training(config, tmp_path / "resumed",
                          resume=tmp_path / "first/checkpoints/final.pt", make_charts=False)
    assert first["training_seeds"] == [100000]
    assert second["training_seeds"] == [100000, 100001]
    assert second["environment_steps_total"] == 20
    assert second["replay_size"] == 10
    config_record = json.loads((tmp_path / "resumed/config.json").read_text())
    assert config_record["resume_mode"].startswith("approximate")


def test_evaluation_rejects_training_seed_overlap():
    agent = DQNAgent(small_config(evaluation_episodes=2))
    agent.checkpoint_metadata = {"training_seeds": [900000]}
    with pytest.raises(ValueError, match="overlap"):
        evaluate_agent(agent)


def test_run_cap_stores_truncation_for_bootstrapping(tmp_path, monkeypatch):
    original = DQNAgent.remember
    observed = []
    def remember(self, state, action, reward, next_state, terminated, truncated):
        observed.append((terminated, truncated))
        return original(self, state, action, reward, next_state, terminated, truncated)
    monkeypatch.setattr(DQNAgent, "remember", remember)
    run_training(small_config(max_environment_steps=1), tmp_path / "cap", make_charts=False)
    assert observed == [(False, True)]


def test_matplotlib_training_charts_are_saved_headlessly(tmp_path):
    from snake_ai.training.charts import create_charts
    from PIL import Image
    output = tmp_path / "charts"
    run_training(small_config(episodes=2, max_environment_steps=30), output, make_charts=False)
    paths = create_charts(output)
    assert len(paths) == 1
    with Image.open(paths[0]) as image:
        assert image.format == "PNG"
        assert image.width >= 1000 and image.height >= 1000
    with pytest.raises(FileExistsError):
        create_charts(output)
