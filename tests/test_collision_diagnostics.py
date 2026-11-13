"""Controlled board states and temporary reference evaluations for diagnostics."""

import csv
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.evaluation.collision_diagnostics import (
    classify_end, diagnose_episode, record_move, run_diagnostics,
)
from snake_ai.game import Direction, GameStatus
from snake_ai.rl import EnvConfig, SnakeEnv
from snake_ai.training.config import TrainingConfig
from snake_ai.training.evaluate import evaluate_agent


@pytest.fixture
def agent_factory():
    def create(q_values=(5., 2., 3.), **config):
        agent = DQNAgent(replace(TrainingConfig(), **config))
        with torch.no_grad():
            agent.online.output_layer.weight.zero_()
            agent.online.output_layer.bias.copy_(torch.tensor(q_values))
        agent.online.eval()
        agent.online.requires_grad_(False)
        return agent
    return create


@pytest.fixture
def env():
    env = SnakeEnv()
    env.reset(seed=7)
    return env


def test_actual_pre_and_post_state_and_q_values_are_copied(env, agent_factory):
    agent = agent_factory()
    env.game.food = (11, 10)
    old_body = [list(position) for position in env.game.snake]
    observation = list(env.observation())
    row = record_move(env, agent, 2, 7)
    assert row["episode"] == 2 and row["environment_seed"] == 7 and row["move"] == 1
    assert row["head_before"] == [10, 10] and row["head_after"] == [11, 10]
    assert row["body_before"] == old_body
    assert row["body_after"] == [list(position) for position in env.game.snake]
    assert row["food_before"] == [11, 10] and row["food_after"] == list(env.game.food)
    assert row["snake_length_before"] == 3 and row["snake_length_after"] == 4
    assert row["score_before"] == 0 and row["score_after"] == 1 and row["reward"] == 10.0
    assert row["observation"] == observation
    assert [row[f"q_{action}"] for action in ("straight", "right", "left")] == [5., 2., 3.]
    assert row["selected_action"] == 0 and row["selected_q"] == 5.
    assert row["direction_before"] == row["movement_direction"] == "RIGHT"
    assert not row["terminated"] and not row["truncated"]
    assert row["end_reason"] is None
    env.game.snake[0] = (0, 0)
    assert row["head_after"] == [11, 10] and row["body_after"][0] == [11, 10]


def test_avoidable_wall_collision_has_safe_alternative_and_q_gap(env, agent_factory):
    env.game.snake = [(19, 10)]
    row = record_move(env, agent_factory(), 1, 7)
    assert [row[f"danger_{action}"] for action in ("straight", "right", "left")] == [True, False, False]
    assert classify_end(row)[0] == "A"
    assert row["selected_dangerous"] and row["safe_actions"] == [1, 2]
    assert row["best_safe_action"] == 2 and row["best_safe_q"] == 3.
    assert row["q_gap_selected_minus_best_safe"] == 2.
    assert row["head_after"] == row["head_before"] == [19, 10]
    assert row["attempted_head"] == [20, 10]  # Projected target, not fabricated post-state.
    assert row["reward"] == -10. and row["game_status_after"] == "lost"


def test_no_immediately_safe_action(env, agent_factory):
    env.game.snake = [(2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3)]
    env.game.direction = Direction.UP
    env.game.food = (0, 0)
    row = record_move(env, agent_factory(), 1, 7)
    assert all(row[f"danger_{name}"] for name in ("straight", "right", "left"))
    assert classify_end(row)[0] == "B"
    assert row["safe_actions"] == [] and row["best_safe_action"] is None
    assert row["best_safe_q"] is None and row["q_gap_selected_minus_best_safe"] is None


@pytest.mark.parametrize("turns", range(4))
def test_predictor_and_relative_actions_all_headings_with_departing_tail(env, agent_factory, turns):
    points = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    def rotate(x, y):
        for _ in range(turns):
            x, y = -y, x
        return (10 + x, 10 + y)
    env.game.snake = [rotate(x, y) for x, y in points]
    env.game.direction = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT][turns]
    env.game.food = (0, 0)
    row = record_move(env, agent_factory(), 1, 7)  # Straight enters the departing tail.
    assert [row[f"danger_{name}"] for name in ("straight", "right", "left")] == [False, True, False]
    assert row["head_after"] == list(rotate(0, -1))
    assert not row["terminated"] and env.game.status is GameStatus.RUNNING
    assert row["safety_matches_observation"] and row["action_mapping_matches_engine"]


def test_safe_negative_q_values_do_not_imply_collision(env, agent_factory):
    env.game.food = (0, 0)
    row = record_move(env, agent_factory(q_values=(-2., -5., -4.)), 1, 7)
    assert row["selected_q"] < 0 and not row["selected_dangerous"]
    assert not row["terminated"]


def test_terminal_win_is_other_not_collision(agent_factory):
    env = SnakeEnv(EnvConfig(width=2, height=2))
    env.reset(seed=7)
    env.game.snake = [(0, 0), (1, 0), (1, 1)]
    env.game.direction = Direction.LEFT
    env.game.food = (0, 1)
    row = record_move(env, agent_factory(q_values=(1., 2., 3.)), 1, 7)
    assert row["terminated"] and not row["truncated"] and row["end_reason"] == "win"
    assert row["food_after"] is None
    assert classify_end(row)[0] == "C"


def test_truncation_is_other_even_though_engine_is_running(agent_factory):
    env = SnakeEnv(EnvConfig(max_steps=1))
    env.reset(seed=7)
    row = record_move(env, agent_factory(), 1, 7)
    assert row["truncated"] and not row["terminated"]
    assert row["game_status_after"] == "running" and row["end_reason"] == "max_steps"
    assert classify_end(row)[0] == "C"


def test_inconsistent_collision_evidence_is_uncertain(env, agent_factory):
    env.game.snake = [(19, 10)]
    row = record_move(env, agent_factory(), 1, 7)
    row["safety_matches_observation"] = False
    assert classify_end(row)[0] == "C"


def test_only_final_thirty_moves_retained_and_fixed_seed_repeats(agent_factory):
    config = EnvConfig(max_steps=40)
    first, second = SnakeEnv(config), SnakeEnv(config)
    agent = agent_factory(q_values=(1., 5., 2.))  # Repeated right turns make a safe small loop.
    episode, moves = diagnose_episode(first, agent, 1, 7)
    assert len(moves) == episode["retained_moves"] == 30
    assert [row["move"] for row in moves] == list(range(11, 41))
    assert episode["steps"] == 40 and episode["truncated"]
    assert (episode, moves) == diagnose_episode(second, agent, 1, 7)


@pytest.fixture
def reference_run(tmp_path, agent_factory):
    agent = agent_factory(evaluation_episodes=3, environment=EnvConfig(max_steps=5))
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    rows, manifest = evaluate_agent(agent)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (reference / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return checkpoint, reference


def test_full_output_schema_reproduction_and_preservation(tmp_path, reference_run):
    checkpoint, reference = reference_run
    protected = [checkpoint, *reference.iterdir()]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    output = tmp_path / "diagnostics"
    result = run_diagnostics(checkpoint, output, reference, make_chart=False)
    assert result["reproduction_verified"] and result["model_weights_unchanged"]
    assert result["training_state_unchanged"] and result["reference_and_checkpoint_files_unchanged"]
    assert result["evaluation"]["episodes"] == 3
    assert result["collision_count"] == 0
    assert result["retained_move_count"] == 15
    assert {path.name for path in output.iterdir()} == {
        "summary.json", "episodes.csv", "final_moves.csv", "collisions.csv", "example_failure.json", "report.md"}
    rows = list(csv.DictReader((output / "final_moves.csv").open(newline="")))
    assert len(rows) == 15
    assert len(json.loads(rows[0]["observation"])) == 11
    assert json.loads(rows[0]["head_before"]) == [10, 10]
    assert len(json.loads(rows[0]["body_before"])) == 3
    assert rows[-1]["truncated"] == "True"
    assert list(csv.DictReader((output / "collisions.csv").open(newline=""))) == []
    examples = json.loads((output / "example_failure.json").read_text())
    assert examples == {"avoidable": None, "no_safe_action": None, "highest_score": None}
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    assert json.loads((output / "summary.json").read_text()) == result
    second = run_diagnostics(checkpoint, tmp_path / "repeat", reference, make_chart=False)
    assert result == second
    assert (output / "final_moves.csv").read_bytes() == (tmp_path / "repeat/final_moves.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_diagnostics(checkpoint, output, reference)


def test_reproduction_mismatch_stops_before_publishing_analysis(tmp_path, reference_run):
    checkpoint, reference = reference_run
    path = reference / "episodes.csv"
    rows = list(csv.DictReader(path.open(newline="")))
    rows[0]["score"] = "99"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="Reproduction failed.*score"):
        run_diagnostics(checkpoint, output, reference, make_chart=False)
    assert not output.exists()


def test_checkpoint_mismatch_stops_before_running(tmp_path, reference_run):
    checkpoint, reference = reference_run
    path = reference / "summary.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_sha256"] = "wrong-hash"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checkpoint hash differs"):
        run_diagnostics(checkpoint, tmp_path / "invalid", reference)


def test_parameters_and_training_state_never_modified(env, agent_factory):
    agent = agent_factory()
    before = {name: value.clone() for name, value in agent.online.state_dict().items()}
    progress = (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())
    diagnose_episode(env, agent, 1, 7)
    assert all(torch.equal(value, agent.online.state_dict()[name]) for name, value in before.items())
    assert progress == (agent.environment_steps, agent.optimiser_steps, agent.epsilon, agent._rng.getstate())
    assert len(agent.replay) == 0


def test_collision_rows_examples_and_chart_are_machine_readable(tmp_path, agent_factory):
    agent = agent_factory(evaluation_episodes=2, environment=EnvConfig(width=4, height=4, max_steps=50))
    checkpoint = tmp_path / "model.pt"
    agent.save(checkpoint)
    rows, manifest = evaluate_agent(agent)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "summary.json").write_text(json.dumps(manifest))
    with (reference / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "diagnostics"
    result = run_diagnostics(checkpoint, output, reference)
    assert result["collision_count"] == 2
    assert result["collision_categories"]["A"]["percentage"] == 100.
    collisions = list(csv.DictReader((output / "collisions.csv").open(newline="")))
    assert len(collisions) == 2 and collisions[0]["collision_category"] == "A"
    assert json.loads(collisions[0]["safe_actions"]) == [1, 2]
    assert json.loads(collisions[0]["body_before"])[0] == json.loads(collisions[0]["head_before"])
    assert float(collisions[0]["q_gap_selected_minus_best_safe"]) == 2.
    assert result["q_value_analysis"]["selected_q_at_collision"]["mean"] == 5.
    examples = json.loads((output / "example_failure.json").read_text())
    assert examples["avoidable"]["final_move"]["selected_action"] == 0
    assert examples["no_safe_action"] is None
    assert (output / "collision_categories.png").read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
