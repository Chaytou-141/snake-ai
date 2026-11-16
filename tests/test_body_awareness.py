"""Static body-only semantics, encoder compatibility and controlled run safety."""
from dataclasses import replace
import copy
import csv
import json
from pathlib import Path
import random

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments import feature_comparison as experiment
from snake_ai.game import Direction
from snake_ai.game.geometry import body_free_cell_distance
from snake_ai.rl import FEATURE_NAMES, EnvConfig, SnakeEnv, feature_names, relative_direction
from snake_ai.training.config import TrainingConfig
from snake_ai.training.train import run_training

PROTOCOL=Path('configs/body_awareness_v1.json')


@pytest.fixture(autouse=True)
def restore_state():
    state=random.getstate();threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):yield
    random.setstate(state);torch.set_num_threads(threads);torch.use_deterministic_algorithms(det)


@pytest.mark.parametrize('direction',list(Direction))
def test_relative_body_rays_nearest_segment_order_and_no_mutation(direction):
    env=SnakeEnv(EnvConfig(width=11,height=11),observation_mode='body16');env.reset(seed=155)
    env.game.snake=[(5,5)];env.game.direction=direction
    for action,distance in [(0,3),(1,2),(2,1),(0,4)]:
        dx,dy=relative_direction(direction,action).value
        env.game.snake.append((5+dx*distance,5+dy*distance))
    before=copy.deepcopy(vars(env.game));obs=env.observation()
    assert list(obs[13:])==pytest.approx([.2,.1,0.])
    after=vars(env.game)
    assert all(before[k]==after[k] for k in before if k!='_rng')
    assert before['_rng'].getstate()==after['_rng'].getstate()
    assert obs.typecode=='f' and obs.itemsize==4 and len(obs)==16
    assert feature_names('body16')==feature_names('food13')+('body_distance_straight','body_distance_right','body_distance_left')


@pytest.mark.parametrize('direction',list(Direction))
def test_walls_do_not_count_as_body_and_clear_ray_returns_one(direction):
    env=SnakeEnv(observation_mode='body16');env.reset(seed=155)
    env.game.snake=[(0,0)];env.game.direction=direction
    assert list(env.observation()[13:])==[1.,1.,1.]
    for action in range(3):assert body_free_cell_distance(env.game,relative_direction(direction,action))==19


def test_body_and_geometry_differ_at_nearby_wall():
    values={}
    for mode in ('original11','distance14','food13','body16'):
        env=SnakeEnv(observation_mode=mode);env.reset(seed=155)
        env.game.snake=[(0,0)];env.game.direction=Direction.LEFT;env.game.food=(5,3)
        values[mode]=env.observation()
    assert list(values['distance14'][11:])==[0.,0.,1.]
    assert list(values['body16'][13:])==[1.,1.,1.]
    assert values['original11']==values['distance14'][:11]==values['food13'][:11]==values['body16'][:11]
    assert values['food13']==values['body16'][:13]
    assert list(values['food13'][11:])==pytest.approx([5/19,3/19])
    assert feature_names()==FEATURE_NAMES


def test_tail_is_static_body_even_when_departure_is_safe():
    env=SnakeEnv(observation_mode='body16');env.reset(seed=155)
    env.game.snake=[(1,1),(1,2),(2,2),(2,1)];env.game.direction=Direction.RIGHT;env.game.food=(8,8)
    obs=env.observation()
    assert obs[0]==0. and obs[13]==0.
    assert body_free_cell_distance(env.game,Direction.RIGHT)==0


@pytest.mark.parametrize('head,body,expected',[
    ((0,0),(6,0),5/6),((0,0),(1,0),0.),((0,0),(0,3),1.),
])
def test_rectangular_board_fixed_normalisation(head,body,expected):
    env=SnakeEnv(EnvConfig(width=7,height=4),observation_mode='body16');env.reset(seed=155)
    env.game.snake=[head,body];env.game.direction=Direction.RIGHT
    obs=env.observation()
    assert obs[13]==pytest.approx(expected)
    assert all(0.<=value<=1. for value in obs[13:])


@pytest.mark.parametrize('mode,dimension',[('original11',11),('distance14',14),('food13',13),('body16',16)])
def test_all_mode_dimensions_checkpoint_and_read_only_evaluation(tmp_path,mode,dimension):
    config=replace(TrainingConfig(),seed=155,observation_mode=mode,evaluation_seed_start=960000,
                   evaluation_episodes=2,environment=EnvConfig(max_steps=3))
    a,b=DQNAgent(config),DQNAgent(config)
    obs,_=SnakeEnv(observation_mode=mode).reset(seed=155)
    assert a.config.input_dim==dimension and a.predict(obs).shape==(3,)
    assert torch.equal(a.predict(obs),b.predict(obs))
    path=tmp_path/'model.pt';a.save(path);before=experiment.digest(path)
    loaded=DQNAgent.load(path)
    assert loaded.config==config and torch.equal(loaded.predict(obs),a.predict(obs))
    rows,summary=experiment.evaluation_rows(loaded,'A' if mode=='food13' else 'B',155)
    assert summary['weights_unchanged'] and summary['input_dim']==dimension and len(rows)==2
    assert experiment.digest(path)==before
    if mode=='body16':assert sum(p.numel() for p in a.online.parameters())==19075


def test_existing_local_5e_models_load_with_original_configs():
    paths=list(Path('results/milestone5e_food_distance/runs').glob('*/training/checkpoints/final.pt'))
    if not paths:pytest.skip('Checkpoints intentionally excluded from Git')
    assert len(paths)==6
    for path in paths:
        before=experiment.digest(path);agent=DQNAgent.load(path)
        obs,_=SnakeEnv(observation_mode=agent.config.observation_mode).reset(seed=155)
        assert agent.config.input_dim in (11,13) and torch.isfinite(agent.predict(obs)).all()
        assert experiment.digest(path)==before


def test_fixed_protocol_paired_budgets_and_unused_seed_blocks():
    p=experiment.load_protocol(PROTOCOL)
    assert p['training_seeds']==[111,122,133] and p['smoke']['training_seed']==155
    prior={7,11,22,33,44,55,66,77,88}
    assert not prior.intersection(p['training_seeds']+[p['smoke']['training_seed']])
    for i,seed in enumerate(p['training_seeds']):
        a=experiment.training_config(p,'A',seed);b=experiment.training_config(p,'B',seed)
        assert a.input_dim==13 and b.input_dim==16
        da,db=a.to_dict(),b.to_dict();da.pop('observation_mode');db.pop('observation_mode');assert da==db
        assert a.max_environment_steps==a.episodes==40000
        assert a.training_seed_start==700000+i*50000
        assert a.evaluation_seed_start==950000 and a.evaluation_episodes==100
    smoke=experiment.training_config(p,'B',155,smoke=True)
    assert smoke.training_seed_start==850000 and smoke.evaluation_seed_start==960000
    assert smoke.max_environment_steps==256 and smoke.input_dim==16
    with pytest.raises(ValueError):experiment.training_config(p,'A',77)


def test_fixed_protocol_mutation_and_overwrite_rejected(tmp_path):
    p=experiment.load_protocol(PROTOCOL);output,metadata=experiment.prepare_output(tmp_path/'experiment',p)
    assert experiment.prepare_output(output,p)==(output,metadata)
    p['base_training_config']['max_environment_steps']=39999
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(p))
    with pytest.raises(ValueError):experiment.load_protocol(changed)
    with pytest.raises(ValueError):experiment.prepare_output(output,p)
    with pytest.raises(ValueError,match='Stored protocol differs'):experiment.prepare_output(Path('results/milestone5e_food_distance'),p)


def test_body_training_step_cap_seed_reproducibility_and_checkpoint(tmp_path):
    p=experiment.load_protocol(PROTOCOL)
    config=replace(experiment.training_config(p,'B',111),episodes=64,max_environment_steps=64,
        learning_starts=8,batch_size=4,environment=EnvConfig(max_steps=1))
    for name in ('first','second'):
        experiment.seed_cpu(155);summary=run_training(config,tmp_path/name,make_charts=False)
        assert summary['status']=='environment_step_cap' and summary['environment_steps_this_run']==64
        assert summary['completed_episodes_this_run']==64 and summary['optimiser_updates_this_run']==57
    assert (tmp_path/'first/episodes.csv').read_bytes()==(tmp_path/'second/episodes.csv').read_bytes()
    a=DQNAgent.load(tmp_path/'first/checkpoints/final.pt');b=DQNAgent.load(tmp_path/'second/checkpoints/final.pt')
    assert all(torch.equal(x,y) for x,y in zip(a.online.parameters(),b.online.parameters()))


def test_saved_run_reused_and_partial_budget_failure_preserved(tmp_path,monkeypatch):
    p=experiment.load_protocol(PROTOCOL);output,meta=experiment.prepare_output(tmp_path/'experiment',p)
    experiment.write_json(output/'validation.json',{'passed':True})
    case=output/'runs/A_seed111';case.mkdir(parents=True);(case/'data.csv').write_text('keep')
    result={'model':'A','training_seed':111,'average_score':1.}
    experiment.write_json(case/'complete.json',{'protocol_sha256':meta['protocol_sha256'],
        'result':result,'artifact_hashes':{'data.csv':experiment.digest(case/'data.csv')}})
    def forbidden(*args,**kwargs):raise AssertionError('Do not repeat successful runs')
    monkeypatch.setattr(experiment,'run_training',forbidden)
    assert experiment.run_one(p,output,'A',111)==result
    (case/'data.csv').write_text('changed')
    with pytest.raises(ValueError):experiment.run_one(p,output,'A',111)
    monkeypatch.setattr(experiment,'run_training',lambda *a,**k:{'status':'environment_step_cap',
        'environment_steps_this_run':39999,'completed_episodes_this_run':500})
    with pytest.raises(RuntimeError,match='Incomplete fixed-step'):
        experiment.run_one(p,output,'B',111)


def test_body_report_and_only_requested_two_charts(tmp_path):
    p=experiment.load_protocol(PROTOCOL);rows=[]
    for seed in p['training_seeds']:
        for model in 'AB':
            rows.append({'model':model,'training_seed':seed,'average_score':2. if model=='B' else 1.,
                'median_score':1.,'minimum_score':0,'maximum_score':4,'score_sd_within_evaluation':1.,
                'average_episode_length':30.,'collision_rate':1.,'truncation_rate':0.,'win_rate':0.,
                'training_environment_steps':40000,'completed_training_episodes':500,
                'optimiser_updates':39001,'training_duration_seconds':60.})
    summary=experiment.compare_results(rows,p['training_seeds'])
    assert summary['mean_paired_difference']==1. and summary['B_higher_all_three_seeds']
    experiment.create_charts(tmp_path,rows,p);experiment.write_body_report(tmp_path,rows,summary,p)
    assert {x.name for x in tmp_path.glob('*.png')}=={'average_score.png','episode_length.png'}
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert '950000-950099' in text and '3/3' in text and 'tail' in text
    with pytest.raises(FileExistsError):experiment.write_body_report(tmp_path,rows,summary,p)
