"""Geometry unit tests and small, fully verified streamed analyses."""

import copy
import csv
from dataclasses import replace
import gzip
import hashlib
import io
import json

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.evaluation.collision_diagnostics import run_diagnostics
from snake_ai.evaluation.trap_analysis import (
    EVENT_FIELDS, analyse_episode, free_cell_distance, measure_state, reachable_component, run_analysis,
)
from snake_ai.game import Direction, SnakeGame
from snake_ai.rl import EnvConfig, SnakeEnv
from snake_ai.training.config import TrainingConfig
from snake_ai.training.evaluate import evaluate_agent


def constant_agent(values=(1., 5., 2.), **kwargs):
    agent = DQNAgent(replace(TrainingConfig(), **kwargs))
    with torch.no_grad():
        agent.online.output_layer.weight.zero_()
        agent.online.output_layer.bias.copy_(torch.tensor(values))
    agent.online.requires_grad_(False)
    agent.online.eval()
    return agent


@pytest.mark.parametrize("turns", range(4))
def test_safe_counts_directions_and_vacating_tail(turns):
    game = SnakeGame(seed=7)
    points = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    def rotate(x, y):
        for _ in range(turns):
            x, y = -y, x
        return (10+x, 10+y)
    game.snake = [rotate(x,y) for x,y in points]
    game.direction = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT][turns]
    game.food = (0,0)
    result = measure_state(game)
    assert result["safe_action_count"] == 2
    assert result["danger"] == [False, True, False]
    assert result["reachable_cells"][0] is not None  # Enters departing tail.
    assert result["reachable_cells"][1] is None  # Immediate body collision.
    assert result["obstacle_free_cells"][0] == 0  # Static distance still sees tail.


@pytest.mark.parametrize("direction, expected", [(Direction.UP, 2), (Direction.RIGHT, 2),
                                                (Direction.DOWN, 2), (Direction.LEFT, 2)])
def test_wall_distance_is_free_cells_not_grid_steps(direction, expected):
    game = SnakeGame(width=5, height=5, seed=7)
    game.snake = [(2,2)]
    assert free_cell_distance(game,direction) == expected
    game.snake = [(0,0)]
    assert free_cell_distance(game, Direction.UP) == 0
    assert free_cell_distance(game, Direction.LEFT) == 0


def test_body_is_nearest_static_obstacle():
    game = SnakeGame(seed=7)
    game.snake = [(10,10),(10,11),(11,11),(12,11),(12,10)]
    assert free_cell_distance(game,Direction.RIGHT) == 1
    assert free_cell_distance(game,Direction.DOWN) == 0


def test_flood_fill_partition_and_start_cell():
    wall = {(2,y) for y in range(5)}
    assert len(reachable_component((0,0),wall,5,5)) == 10
    assert len(reachable_component((4,0),wall,5,5)) == 10
    assert reachable_component((2,0),wall,5,5) == set()
    assert reachable_component((-1,0),wall,5,5) == set()
    assert reachable_component((0,0),{(0,1),(1,0)},2,2) == {(0,0)}


def test_candidate_growth_and_tail_match_actual_first_step_without_sampling(monkeypatch):
    game = SnakeGame(width=5,height=5,seed=7)
    game.snake = [(2,2),(1,2),(0,2)]
    game.food = (3,2)
    result = measure_state(game)
    assert result["reachable_cells"] == [22,23,23]
    clone = copy.deepcopy(game)
    clone.step(Direction.RIGHT)  # Known food is eaten; its complete old body remains.
    assert len(reachable_component(clone.snake[0],set(clone.snake[1:]),5,5)) == 22
    def forbidden():
        raise AssertionError("Geometric diagnostics must never spawn hypothetical food")
    monkeypatch.setattr(game,"_spawn_food",forbidden)
    assert measure_state(game) == result


def test_geometry_does_not_mutate_state_or_rng():
    game = SnakeGame(seed=42)
    before = (game.snake.copy(),game.direction,game.food,game.score,game.status,game._rng.getstate())
    assert measure_state(game) == measure_state(game)
    assert before == (game.snake,game.direction,game.food,game.score,game.status,game._rng.getstate())


def test_static_area_one_is_not_a_dynamic_death_prediction():
    # Recorded geometry from seed 900045, move 152; no model/training dependency.
    game = SnakeGame(seed=7)
    game.snake = [(6,13),(6,14),(7,14),(7,15),(7,16),(7,17),(7,18),(8,18),
                  (8,17),(8,16),(8,15),(8,14),(8,13),(8,12),(7,12),(7,13)]
    game.direction = Direction.UP
    game.food = (16,0)
    assert measure_state(game)["reachable_cells"] == [384,1,384]
    game.step(Direction.RIGHT)  # Enter the departing tail cell.
    assert not game.would_collide(Direction.UP)  # The next tail cell also departs.
    game.step(Direction.UP)
    assert game.status.value == "running"


def run_episode(env,agent,seed=7):
    trace,event_file = io.StringIO(),io.StringIO()
    writer=csv.DictWriter(event_file,fieldnames=EVENT_FIELDS)
    writer.writeheader()
    result,history=analyse_episode(env,agent,1,seed,trace,writer,low_space_cells=20,substantial_gap_cells=20)
    return result,history,[json.loads(line) for line in trace.getvalue().splitlines()]


def test_full_episode_trace_termination_and_no_weight_changes():
    agent=constant_agent(values=(5.,1.,2.))
    before=[p.clone() for p in agent.online.parameters()]
    episode,history,trace=run_episode(SnakeEnv(),agent)
    assert episode["end_reason"] == "collision" and episode["terminated"]
    assert episode["first_zero_safe_move"] is None  # Fixed test policy chooses a wall with alternatives.
    assert trace[-1]["safe_action_count"] == 2
    assert trace[-1]["reachable_cells"][0] is None
    assert len(trace) == episode["steps"] == len(history["safe_before"])
    assert trace[-1]["safe_action_count_after"] is None
    assert all(torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    assert agent.environment_steps == agent.optimiser_steps == 0


def test_truncated_episode_and_deterministic_stream():
    agent=constant_agent()
    first=run_episode(SnakeEnv(EnvConfig(max_steps=40)),agent)
    second=run_episode(SnakeEnv(EnvConfig(max_steps=40)),agent)
    assert first == second
    episode,history,trace=first
    assert len(trace) == 40  # Unlike 5B, the complete episode is retained on disk.
    assert episode["truncated"] and not episode["terminated"]
    assert trace[-1]["safe_action_count_after"] is not None


def test_full_grid_win_uses_legal_geometry_not_collision():
    # A fixture environment arranges a known last-food move on reset.
    class WinningEnv(SnakeEnv):
        def reset(self,seed=None):
            super().reset(seed)
            self.game.snake=[(0,0),(1,0),(1,1)]
            self.game.direction=Direction.LEFT
            self.game.food=(0,1)
            return self.observation(),self._info()
    episode,history,trace=run_episode(WinningEnv(EnvConfig(width=2,height=2)),
                                    constant_agent(values=(1.,2.,5.)))
    assert episode["terminated"] and episode["end_reason"] == "win"
    assert trace[0]["reachable_cells"][2] == 1
    assert trace[0]["score_after"] == 1


@pytest.fixture
def references(tmp_path):
    agent=constant_agent(evaluation_episodes=3,environment=EnvConfig(max_steps=8))
    checkpoint=tmp_path/"model.pt"
    agent.save(checkpoint)
    rows,manifest=evaluate_agent(agent)
    manifest["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference=tmp_path/"reference"
    reference.mkdir()
    (reference/"summary.json").write_text(json.dumps(manifest))
    with (reference/"episodes.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    ref5b=tmp_path/"5b"
    run_diagnostics(checkpoint,ref5b,reference,make_chart=False)
    return checkpoint,reference,ref5b


def test_complete_pipeline_outputs_reproduction_and_preservation(tmp_path,references):
    checkpoint,reference,ref5b=references
    protected=[checkpoint,*reference.iterdir(),*ref5b.iterdir()]
    hashes={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    output=tmp_path/"analysis"
    summary=run_analysis(checkpoint,output,reference,ref5b)
    assert summary["reproduces_milestone4_and_5b"]
    assert summary["evaluation"]["episodes"] == 3
    assert summary["analysis"]["movement_count"] == 24
    assert not (output/"INCOMPLETE").exists()
    with gzip.open(output/"state_traces.jsonl.gz","rt") as handle:
        trace=[json.loads(line) for line in handle]
    assert len(trace)==24 and len(trace[0]["body_before"])==3
    assert len(trace[0]["reachable_cells"])==3
    assert all((output/name).exists() for name in ["summary.json","episodes.csv","trap_events.csv",
        "report.md","safe_actions.png","reachable_space.png","action_comparison.png"])
    assert hashes=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    repeated=run_analysis(checkpoint,tmp_path/"repeat",reference,ref5b,make_charts=False)
    assert summary==repeated
    assert (output/"state_traces.jsonl.gz").read_bytes()==(tmp_path/"repeat/state_traces.jsonl.gz").read_bytes()
    with pytest.raises(FileExistsError):
        run_analysis(checkpoint,output,reference,ref5b)


def test_failed_reproduction_leaves_incomplete_marker_not_report(tmp_path,references):
    checkpoint,reference,ref5b=references
    path=reference/"episodes.csv"
    rows=list(csv.DictReader(path.open(newline="")))
    rows[0]["score"]="99"
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    output=tmp_path/"mismatch"
    with pytest.raises(ValueError,match="Reproduction failed"):
        run_analysis(checkpoint,output,reference,ref5b,make_charts=False)
    assert (output/"INCOMPLETE").exists()
    assert not (output/"report.md").exists()
