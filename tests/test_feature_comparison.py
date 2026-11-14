"""Feature semantics, compatibility, pairing, and safe experiment orchestration."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments.feature_comparison import (
    DEFAULT_PROTOCOL, compare_results, digest, evaluation_rows, load_protocol,
    prepare_output, run_one, run_smoke, training_config, write_json,
)
from snake_ai.game import Direction
from snake_ai.rl import FEATURE_NAMES, EnvConfig, SnakeEnv, feature_names
from snake_ai.training.config import TrainingConfig


@pytest.fixture(autouse=True)
def restore_global_state():
    python_state=random.getstate()
    threads=torch.get_num_threads()
    deterministic=torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(python_state)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def test_default_eleven_features_unchanged_and_prefix_identical():
    original=SnakeEnv()
    extended=SnakeEnv(observation_mode="distance14")
    obs,_=original.reset(seed=77)
    extra,_=extended.reset(seed=77)
    assert len(obs)==11 and len(extra)==14
    assert obs==extra[:11]
    assert set(obs)<={0.,1.}
    for action in [1,1,1,1,0,2]:
        a=original.step(action)
        b=extended.step(action)
        assert a[0]==b[0][:11]
        assert a[1:]==b[1:]


def test_exact_feature_order_and_distance_normalisation():
    env=SnakeEnv(EnvConfig(width=5,height=5),observation_mode="distance14")
    env.reset(seed=77)
    env.game.snake=[(2,2)]
    env.game.direction=Direction.RIGHT
    assert feature_names()==FEATURE_NAMES
    assert feature_names("distance14")==FEATURE_NAMES+("distance_straight","distance_right","distance_left")
    assert list(env.observation()[11:])==[.5,.5,.5]  # 2 free cells / max(5,5)-1.
    env.game.snake=[(0,0)]
    assert list(env.observation()[11:])==[1.,1.,0.]


@pytest.mark.parametrize("direction",list(Direction))
def test_all_relative_distance_directions(direction):
    env=SnakeEnv(EnvConfig(width=6,height=5),observation_mode="distance14")
    env.reset(seed=77)
    env.game.snake=[(1,3)]
    env.game.direction=direction
    # Width is 6, so divisor is 5. Expected raw counts are derived from the grid.
    raw={Direction.UP:[3,4,1],Direction.RIGHT:[4,1,3],
         Direction.DOWN:[1,1,4],Direction.LEFT:[1,3,1]}[direction]
    assert list(env.observation()[11:])==pytest.approx([count/5 for count in raw])


def test_static_tail_distance_is_zero_but_departing_tail_remains_safe():
    env=SnakeEnv(observation_mode="distance14")
    env.reset(seed=77)
    env.game.snake=[(2,2),(2,3),(3,3),(3,2)]
    env.game.direction=Direction.UP
    env.game.food=(0,0)
    obs=env.observation()
    assert obs[1]==0. and obs[12]==0.  # Relative right enters departing tail.
    assert env.step(1)[2:4]==(False,False)
    assert env.game.snake[0]==(3,2)


def test_distance_bounds_and_no_rng_or_state_mutation():
    env=SnakeEnv(observation_mode="distance14")
    env.reset(seed=77)
    before=(env.game.snake.copy(),env.game.food,env.game._rng.getstate())
    obs=env.observation()
    assert all(0<=value<=1 for value in obs[11:])
    assert before==(env.game.snake,env.game.food,env.game._rng.getstate())
    assert obs.typecode=="f"
    with pytest.raises(ValueError):
        SnakeEnv(observation_mode="unknown")


@pytest.mark.parametrize("mode,dimension,parameters",[("original11",11,18435),("distance14",14,18819)])
def test_model_replay_training_and_checkpoint_encoder_match(tmp_path,mode,dimension,parameters):
    config=replace(TrainingConfig(),seed=77,observation_mode=mode,learning_starts=2,batch_size=2)
    first,second=DQNAgent(config),DQNAgent(config)
    env=SnakeEnv(observation_mode=mode)
    obs,_=env.reset(seed=77)
    assert first.predict(obs).shape==(3,)
    assert torch.equal(first.predict(obs),second.predict(obs))
    assert sum(p.numel() for p in first.online.parameters())==parameters
    for _ in range(2):
        first.remember(obs,0,0.,obs,False,False)
    assert first.replay.sample(2).states.shape==(2,dimension)
    assert first.train_step() is not None
    checkpoint=tmp_path/"model.pt"
    first.save(checkpoint)
    loaded=DQNAgent.load(checkpoint)
    assert loaded.config.input_dim==dimension and loaded.config.observation_mode==mode
    assert torch.equal(first.predict(obs),loaded.predict(obs))


def test_legacy_checkpoint_without_new_config_field_still_loads(tmp_path):
    config=replace(TrainingConfig(),seed=77)
    assert "observation_mode" not in config.to_dict()
    agent=DQNAgent(config)
    path=tmp_path/"legacy.pt"
    agent.save(path)
    payload=torch.load(path,weights_only=True)
    assert "observation_mode" not in payload["config"]
    loaded=DQNAgent.load(path)
    assert loaded.config.observation_mode=="original11" and loaded.config.input_dim==11


def test_existing_local_seed7_checkpoint_is_read_only_compatible():
    checkpoint=Path(__file__).resolve().parents[1]/"results/milestone4_initial/checkpoints/final.pt"
    if not checkpoint.exists():
        pytest.skip("Historical checkpoint intentionally excluded from Git")
    before=digest(checkpoint)
    agent=DQNAgent.load(checkpoint)
    obs,_=SnakeEnv().reset(seed=800000)
    assert agent.config.input_dim==11 and torch.isfinite(agent.predict(obs)).all()
    assert digest(checkpoint)==before


@pytest.mark.parametrize("mode",["original11","distance14"])
def test_greedy_evaluation_correct_encoder_and_no_updates(mode):
    config=replace(TrainingConfig(),seed=77,observation_mode=mode,evaluation_episodes=2,
                   evaluation_seed_start=920000,environment=EnvConfig(max_steps=3))
    agent=DQNAgent(config)
    before=[p.clone() for p in agent.online.parameters()]
    progress=(agent.environment_steps,agent.optimiser_steps,agent.epsilon,agent._rng.getstate())
    rows,summary=evaluation_rows(agent,"A" if mode=="original11" else "B",77)
    assert len(rows)==2
    assert [r["evaluation_seed"] for r in rows]==[920000,920001]
    assert summary["input_dim"]==config.input_dim
    assert summary["weights_unchanged"]
    assert all(torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    assert progress==(agent.environment_steps,agent.optimiser_steps,agent.epsilon,agent._rng.getstate())


def test_fixed_protocol_and_paired_food_schedule():
    protocol=load_protocol()
    assert protocol["training_seeds"]==[11,22,33]
    for seed in protocol["training_seeds"]:
        a=training_config(protocol,"A",seed)
        b=training_config(protocol,"B",seed)
        assert a.training_seed_start==b.training_seed_start==200000+1000*seed
        assert a.evaluation_seed_start==b.evaluation_seed_start==910000
        assert a.input_dim==11 and b.input_dim==14
        assert a.episodes==b.episodes==500
        assert a.max_environment_steps==b.max_environment_steps==100000
        env_a=SnakeEnv(a.environment,a.rewards,observation_mode=a.observation_mode)
        env_b=SnakeEnv(b.environment,b.rewards,observation_mode=b.observation_mode)
        assert env_a.reset(seed=a.training_seed_start)[0]==env_b.reset(seed=b.training_seed_start)[0][:11]
    with pytest.raises(ValueError):
        training_config(protocol,"A",7)


def test_changed_protocol_settings_are_rejected(tmp_path):
    protocol=load_protocol()
    protocol["base_training_config"]["episodes"]=501
    path=tmp_path/"changed.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match="budgets"):
        load_protocol(path)


def test_aggregation_uses_three_training_seed_means_and_pairs():
    rows=[{"model":model,"training_seed":seed,"average_score":score}
          for model,scores in (("A",[10.,20.,30.]),("B",[15.,18.,39.]))
          for seed,score in zip((11,22,33),scores)]
    summary=compare_results(rows)
    assert summary["variants"]["A"]["mean_of_training_seed_means"]==20.
    assert summary["variants"]["A"]["sd_across_training_seed_means"]==10.
    assert summary["paired_differences_B_minus_A"]=={"11":5.,"22":-2.,"33":9.}
    assert summary["mean_paired_difference"]==4.
    assert not summary["B_higher_all_three_seeds"] and not summary["direction_consistent_nonzero"]
    with pytest.raises(ValueError):
        compare_results(rows[:-1])


def test_output_freeze_and_no_overwrite(tmp_path):
    protocol=load_protocol()
    output=tmp_path/"experiment"
    path,metadata=prepare_output(output,protocol)
    before=(path/"protocol.json").read_bytes()
    assert prepare_output(output,protocol)==(path,metadata)
    changed=dict(protocol);changed["hypothesis"]="changed after results"
    with pytest.raises(ValueError):
        prepare_output(output,changed)
    assert (path/"protocol.json").read_bytes()==before
    unrelated=tmp_path/"unrelated";unrelated.mkdir()
    (unrelated/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_output(unrelated,protocol)


def test_completed_run_is_preserved_without_training(tmp_path,monkeypatch):
    protocol=load_protocol();output,metadata=prepare_output(tmp_path/"experiment",protocol)
    write_json(output/"validation.json",{"passed":True})
    case=output/"runs/A_seed11";case.mkdir(parents=True)
    (case/"existing.csv").write_text("verified")
    result={"model":"A","training_seed":11,"average_score":1.}
    write_json(case/"complete.json",{"protocol_sha256":metadata["protocol_sha256"],"result":result,
        "artifact_hashes":{"existing.csv":digest(case/"existing.csv")}})
    def forbidden(*args,**kwargs):
        raise AssertionError("A successful run must not be repeated")
    monkeypatch.setattr("snake_ai.experiments.feature_comparison.run_training",forbidden)
    assert run_one(protocol,output,"A",11)==result
    (case/"existing.csv").write_text("changed")
    with pytest.raises(ValueError):
        run_one(protocol,output,"A",11)


def test_final_runs_require_development_validation(tmp_path):
    with pytest.raises(RuntimeError,match="smoke test"):
        run_one(load_protocol(),tmp_path/"experiment","A",11)


def test_comparison_charts_and_report_from_synthetic_measurements(tmp_path):
    from snake_ai.experiments.feature_comparison import create_charts,write_report
    rows=[]
    for model in ("A","B"):
        for seed in (11,22,33):
            path=tmp_path/"runs"/f"{model}_seed{seed}"/"training"
            path.mkdir(parents=True)
            (path/"episodes.csv").write_text("episode_complete,score\nTrue,1\nTrue,2\n")
            rows.append({"model":model,"training_seed":seed,"average_score":float(seed),
                "completed_training_episodes":2,"training_environment_steps":30,"optimiser_updates":2,
                "median_score":1.,"score_sd_within_evaluation":.5,"average_episode_length":15.,
                "collision_rate":1.,"truncation_rate":0.,"win_rate":0.})
    create_charts(tmp_path,rows)
    write_report(tmp_path,rows,compare_results(rows))
    assert {path.name for path in tmp_path.glob("*.png")}=={
        "average_score.png","episode_length.png","training_performance.png","performance_variability.png"}
    assert "Paired B" in (tmp_path/"report.md").read_text(encoding="utf-8")
