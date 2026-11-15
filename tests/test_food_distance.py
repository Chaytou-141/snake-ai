"""Food13 feature semantics, matched budgets and safe protocol execution."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import random

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments import feature_comparison as experiment
from snake_ai.game import Direction
from snake_ai.rl import FEATURE_NAMES, EnvConfig, SnakeEnv, feature_names
from snake_ai.training.config import TrainingConfig
from snake_ai.training.train import run_training

PROTOCOL=Path("configs/food_distance_v1.json")


@pytest.fixture(autouse=True)
def restore_state():
    state=random.getstate(); threads=torch.get_num_threads()
    deterministic=torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(state);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize("head,food,expected",[
    ((0,0),(19,19),(1.,1.)), ((19,19),(0,0),(-1.,-1.)),
    ((10,10),(10,0),(0.,-10/19)), ((10,10),(0,10),(-10/19,0.)),
    ((0,19),(19,0),(1.,-1.)), ((19,0),(0,19),(-1.,1.)),
    ((10,10),(10,10),(0.,0.)), ((10,10),None,(0.,0.)),
])
@pytest.mark.parametrize("direction",list(Direction))
def test_food_offsets_and_original_encoders_unchanged(head,food,expected,direction):
    observations={}
    for mode in ("original11","distance14","food13"):
        env=SnakeEnv(observation_mode=mode);env.reset(seed=88)
        env.game.snake=[head];env.game.direction=direction;env.game.food=food
        before=env.game._rng.getstate()
        obs=env.observation();observations[mode]=obs
        assert obs.typecode=="f" and obs.itemsize==4
        assert env.game._rng.getstate()==before
    assert observations['original11']==observations['food13'][:11]==observations['distance14'][:11]
    assert list(observations['food13'][11:])==pytest.approx(expected)
    assert all(-1<=x<=1 for x in observations['food13'][11:])
    assert len(observations['distance14'])==14


def test_rectangular_board_normalisation_and_feature_order():
    env=SnakeEnv(EnvConfig(width=7,height=4),observation_mode='food13');env.reset(seed=88)
    env.game.snake=[(1,2)];env.game.food=(5,0)
    assert list(env.observation()[11:])==pytest.approx([4/6,-2/3])
    assert feature_names('food13')==FEATURE_NAMES+('food_dx','food_dy')
    assert feature_names()==FEATURE_NAMES


@pytest.mark.parametrize('mode,dimension',[('original11',11),('distance14',14),('food13',13)])
def test_dimensions_checkpoints_reproducibility_and_greedy_evaluation(tmp_path,mode,dimension):
    config=replace(TrainingConfig(),seed=88,observation_mode=mode,evaluation_seed_start=940000,
                   evaluation_episodes=2,environment=EnvConfig(max_steps=3))
    a,b=DQNAgent(config),DQNAgent(config)
    obs,_=SnakeEnv(observation_mode=mode).reset(seed=88)
    assert a.config.input_dim==dimension and torch.equal(a.predict(obs),b.predict(obs))
    path=tmp_path/'final.pt';a.save(path);before=experiment.digest(path)
    loaded=DQNAgent.load(path)
    assert loaded.config==config and torch.equal(a.predict(obs),loaded.predict(obs))
    if mode!='original11':
        with pytest.raises(ValueError):loaded.predict(obs[:11])
    rows,metrics=experiment.evaluation_rows(loaded,'A' if dimension==11 else 'B',88)
    assert metrics['input_dim']==dimension and metrics['weights_unchanged']
    assert len(rows)==2 and experiment.digest(path)==before


def test_predeclared_equal_move_protocol_and_disjoint_seed_blocks():
    p=experiment.load_protocol(PROTOCOL)
    prior_end=None
    for seed in (44,55,66):
        a=experiment.training_config(p,'A',seed);b=experiment.training_config(p,'B',seed)
        assert a.input_dim==11 and b.input_dim==13
        da,db=a.to_dict(),b.to_dict();db.pop('observation_mode');assert da==db
        assert a.max_environment_steps==a.episodes==40000
        assert a.training_seed_start==300000+seed*5000
        assert prior_end is None or prior_end<a.training_seed_start
        prior_end=a.training_seed_start+39999
        assert prior_end<930000
        assert a.evaluation_seed_start==930000 and a.evaluation_episodes==100
    assert p['smoke']['evaluation_seed_start']==940000
    with pytest.raises(ValueError):experiment.training_config(p,'A',7)


@pytest.mark.parametrize('mode',['original11','food13'])
def test_budget_is_steps_even_with_short_episodes_and_seeded_training(tmp_path,mode):
    p=experiment.load_protocol(PROTOCOL)
    config=replace(experiment.training_config(p,'A' if mode=='original11' else 'B',44),
        episodes=64,max_environment_steps=64,learning_starts=8,batch_size=4,
        environment=EnvConfig(max_steps=1))
    runs=[]
    for name in ('first','second'):
        experiment.seed_cpu(88)
        summary=run_training(config,tmp_path/name,make_charts=False);runs.append(summary)
        assert summary['status']=='environment_step_cap'
        assert summary['environment_steps_this_run']==64 and summary['completed_episodes_this_run']==64
        assert summary['optimiser_updates_this_run']==57
    assert (tmp_path/'first/episodes.csv').read_bytes()==(tmp_path/'second/episodes.csv').read_bytes()
    a=DQNAgent.load(tmp_path/'first/checkpoints/final.pt');b=DQNAgent.load(tmp_path/'second/checkpoints/final.pt')
    assert all(torch.equal(x,y) for x,y in zip(a.online.parameters(),b.online.parameters()))


def test_food_protocol_freeze_and_historical_output_protection(tmp_path):
    p=experiment.load_protocol(PROTOCOL);output,metadata=experiment.prepare_output(tmp_path/'new',p)
    assert experiment.prepare_output(output,p)==(output,metadata)
    p['base_training_config']['max_environment_steps']=40001
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(p))
    with pytest.raises(ValueError):experiment.load_protocol(changed)
    with pytest.raises(ValueError):experiment.prepare_output(output,p)
    with pytest.raises(FileExistsError):experiment.prepare_output(Path('results/milestone4_initial'),p)


def test_pair_aggregation_and_no_repeat_of_completed_training(tmp_path,monkeypatch):
    p=experiment.load_protocol(PROTOCOL)
    rows=[dict(model=m,training_seed=s,average_score=float(s)+(2 if m=='B' else 0))
          for s in (44,55,66) for m in 'AB']
    result=experiment.compare_results(rows,p['training_seeds'])
    assert result['mean_paired_difference']==2 and result['B_higher_all_three_seeds']
    output,meta=experiment.prepare_output(tmp_path/'experiment',p)
    experiment.write_json(output/'validation.json',{'passed':True})
    case=output/'runs/A_seed44';case.mkdir(parents=True)
    (case/'data.csv').write_text('preserve')
    experiment.write_json(case/'complete.json',{'protocol_sha256':meta['protocol_sha256'],
        'result':rows[0],'artifact_hashes':{'data.csv':experiment.digest(case/'data.csv')}})
    def forbidden(*args,**kwargs):raise AssertionError('Successful runs must not repeat')
    monkeypatch.setattr(experiment,'run_training',forbidden)
    assert experiment.run_one(p,output,'A',44)==rows[0]
    (case/'data.csv').write_text('changed')
    with pytest.raises(ValueError):experiment.run_one(p,output,'A',44)


def test_incomplete_fixed_step_run_is_preserved_not_evaluated(tmp_path,monkeypatch):
    p=experiment.load_protocol(PROTOCOL);output,_=experiment.prepare_output(tmp_path/'experiment',p)
    experiment.write_json(output/'validation.json',{'passed':True})
    monkeypatch.setattr(experiment,'run_training',lambda *a,**k:{'status':'episode_budget',
        'environment_steps_this_run':39999,'completed_episodes_this_run':500})
    with pytest.raises(RuntimeError,match='Incomplete fixed-step'):
        experiment.run_one(p,output,'A',44)


def test_three_food_charts_and_report_from_saved_measurements(tmp_path):
    p=experiment.load_protocol(PROTOCOL);rows=[]
    for seed in (44,55,66):
        for model in 'AB':
            directory=tmp_path/f'runs/{model}_seed{seed}/training';directory.mkdir(parents=True)
            (directory/'episodes.csv').write_text('episode_complete,score,environment_steps\nTrue,1,20\nTrue,2,40\nFalse,0,50\n')
            rows.append({'model':model,'training_seed':seed,'average_score':2.,'median_score':2.,
                'minimum_score':0,'maximum_score':4,'score_sd_within_evaluation':1.,
                'average_episode_length':30.,'collision_rate':1.,'truncation_rate':0.,'win_rate':0.,
                'training_environment_steps':40000,'completed_training_episodes':500,
                'optimiser_updates':39001,'training_duration_seconds':60.})
    summary=experiment.compare_results(rows,p['training_seeds'])
    experiment.create_charts(tmp_path,rows,p);experiment.write_food_report(tmp_path,rows,summary)
    assert {x.name for x in tmp_path.glob('*.png')}=={'average_score.png','episode_length.png','training_performance.png'}
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert '930000-930099' in text and '0/3' in text and 'exploratory' in text
    with pytest.raises(FileExistsError):experiment.write_food_report(tmp_path,rows,summary)
