"""Model contracts and actual gradient/parameter changes; no RL training."""

import json
import math
import subprocess
import sys

import pytest
import torch
from torch import nn

from snake_ai.models import DQNNetwork
from snake_ai.rl import SnakeEnv


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    # Seeding tests must not affect other tests or depend on their run order.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield


def test_single_environment_observation_and_action():
    env = SnakeEnv()
    observation, _ = env.reset(seed=42)
    inputs = torch.tensor(list(observation), dtype=torch.float32)
    model = DQNNetwork()
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (3,)
    assert outputs.dtype == torch.float32
    assert outputs.device.type == "cpu"
    assert torch.isfinite(outputs).all()
    action = int(outputs.argmax().item())
    _, reward, terminated, truncated, info = env.step(action)
    assert action in (0, 1, 2)
    assert math.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["steps"] == 1


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batch_output_shape_and_finiteness(batch_size):
    model = DQNNetwork()
    observations = torch.randint(0, 2, (batch_size, 11)).float()
    outputs = model(observations)
    assert outputs.shape == (batch_size, 3)
    assert torch.isfinite(outputs).all()


def test_custom_dimensions():
    model = DQNNetwork(input_dim=5, hidden_size=16, output_dim=2)
    assert model(torch.zeros(5)).shape == (2,)
    assert model(torch.zeros(4, 5)).shape == (4, 2)
    assert sum(model.parameter_counts().values()) == (5 * 16 + 16) + (16 * 16 + 16) + (16 * 2 + 2)


@pytest.mark.parametrize("inputs", [torch.zeros(10), torch.zeros(2, 12),
                                     torch.tensor(1.0), torch.zeros(2, 3, 11)])
def test_invalid_input_dimensions(inputs):
    with pytest.raises(ValueError, match="Expected shape"):
        DQNNetwork()(inputs)


def test_non_tensor_and_integer_inputs():
    model = DQNNetwork()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model([0.0] * 11)
    with pytest.raises(TypeError, match="floating point"):
        model(torch.zeros(11, dtype=torch.int64))


@pytest.mark.parametrize("kwargs", [{"input_dim": 0}, {"hidden_size": -1},
                                     {"output_dim": True}, {"input_dim": 11.0}])
def test_invalid_model_dimensions(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        DQNNetwork(**kwargs)


def test_raw_outputs_can_be_negative_and_do_not_sum_to_one():
    model = DQNNetwork()
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.copy_(torch.tensor([-2.0, 0.0, 4.0]))
    assert torch.equal(model(torch.zeros(11)), torch.tensor([-2.0, 0.0, 4.0]))


def test_layer_and_total_trainable_parameter_counts():
    model = DQNNetwork()
    assert isinstance(model, nn.Module)
    counts = model.parameter_counts()
    assert counts == {"input_layer": 1536, "hidden_layer": 16512, "output_layer": 387}
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert sum(counts.values()) == actual == 18435


def test_gradients_are_finite_and_optimiser_changes_parameters():
    model = DQNNetwork()
    observations = torch.randint(0, 2, (4, 11)).float()
    targets = torch.tensor([[1.0, -1.0, 0.5]] * 4)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    loss = nn.MSELoss()(model(observations), targets)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimiser.zero_grad()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in model.parameters())
    optimiser.step()
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, model.parameters()))


def test_same_seed_same_initialisation_and_predictions():
    env = SnakeEnv()
    first_obs, _ = env.reset(seed=42)
    torch.manual_seed(7)
    first = DQNNetwork()
    second_obs, _ = env.reset(seed=42)
    torch.manual_seed(7)
    second = DQNNetwork()
    assert first_obs == second_obs
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    inputs = torch.tensor(list(first_obs), dtype=torch.float32)
    assert torch.equal(first(inputs), second(inputs))


def test_headless_network_demo_repeats():
    from snake_ai.demos.network import run_demo
    first = run_demo()
    assert first == run_demo()
    assert "UNTRAINED" in first["description"]
    assert len(first["observation"]) == 11
    assert len(first["q_values"]) == 3
    assert first["selected_action"] == max(range(3), key=first["q_values"].__getitem__)


def test_educational_step_has_gradients_and_changes_weights():
    from snake_ai.demos.optimisation import run_demo
    result = run_demo()
    assert result == run_demo()
    assert result["optimiser_steps"] == 1
    assert result["gradient_l2_norm"] > 0
    assert result["changed_parameter_tensors"] > 0
    assert math.isfinite(result["loss_before"]) and math.isfinite(result["loss_after"])
    # No generic assertion that every possible optimiser step reduces the loss.


@pytest.mark.parametrize("module", ["network", "optimisation"])
def test_demo_command_runs_without_loading_pygame(module):
    code = (f"import runpy,sys; runpy.run_module('snake_ai.demos.{module}', run_name='__main__'); "
            "assert 'pygame' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["device"] == "cpu"
