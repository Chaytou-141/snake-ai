"""Seeded policy and repeatable baseline artifact checks."""

import csv
import json
import random

import pytest

from snake_ai.evaluate import evaluate, save_results
from snake_ai.rl import EnvConfig, RandomAgent


def test_random_agent_seed_and_action_space():
    before = random.getstate()
    first, second = RandomAgent(7), RandomAgent(7)
    actions = [first.act() for _ in range(100)]
    assert actions == [second.act() for _ in range(100)]
    assert set(actions) == {0, 1, 2}
    first.reset(seed=7)
    assert actions == [first.act() for _ in range(100)]
    assert random.getstate() == before


def test_evaluation_repeatability_and_summary(tmp_path):
    args = dict(episodes=5, seed_start=2, agent_seed_start=100,
                config=EnvConfig(max_steps=3))
    rows, manifest = evaluate(**args)
    assert (rows, manifest) == evaluate(**args)
    assert manifest["environment_seeds"] == [2, 3, 4, 5, 6]
    assert manifest["agent_seeds"] == [100, 101, 102, 103, 104]
    summary = manifest["summary"]
    assert summary["episodes"] == 5
    assert summary["average_episode_length"] == sum(row["steps"] for row in rows) / 5
    assert summary["average_score"] == sum(row["score"] for row in rows) / 5
    assert summary["median_score"] == sorted(row["score"] for row in rows)[2]
    assert summary["maximum_score"] == max(row["score"] for row in rows)
    assert summary["minimum_score"] == min(row["score"] for row in rows)
    assert summary["collision_rate"] + summary["truncation_rate"] + summary["win_rate"] == 1
    output = tmp_path / "baseline.csv"
    save_results(output, rows, manifest)
    with output.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == 5
    assert [int(row["steps"]) for row in loaded] == [row["steps"] for row in rows]
    assert json.loads(output.with_suffix(".json").read_text()) == manifest
    old_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_results(output, rows, manifest)
    assert output.read_bytes() == old_bytes


def test_collision_and_truncation_rates_are_measured_from_episodes():
    rows, report = evaluate(episodes=20, config=EnvConfig(max_steps=15))
    assert report["summary"]["collision_rate"] == sum(row["end_reason"] == "collision" for row in rows) / 20
    assert report["summary"]["truncation_rate"] == sum(row["truncated"] for row in rows) / 20
    assert all(row["terminated"] != row["truncated"] for row in rows)


def test_invalid_episode_count():
    with pytest.raises(ValueError):
        evaluate(episodes=0)
