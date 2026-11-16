"""Snapshot side effects, continuous budgets, paired analysis and preservation."""

from dataclasses import replace
import importlib
import json
import random

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments import training_duration as duration
from snake_ai.experiments.feature_comparison import digest, evaluation_rows, seed_cpu, write_json
from snake_ai.rl import EnvConfig

train = importlib.import_module('snake_ai.training.train')


@pytest.fixture(autouse=True)
def restore_state():
    state=random.getstate();threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        yield
    random.setstate(state);torch.set_num_threads(threads);torch.use_deterministic_algorithms(det)


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:
        assert a==b


def test_snapshot_saving_preserves_entire_training_trajectory(tmp_path,monkeypatch):
    config=replace(duration.config_for(duration.load_protocol(),177),episodes=96,max_environment_steps=96,
                   learning_starts=8,batch_size=4,replay_capacity=64,target_update_interval=7)
    agents=[];envs=[]
    agent_type=train.DQNAgent;env_type=train.SnakeEnv
    def agent_factory(*args,**kwargs):
        agent=agent_type(*args,**kwargs);agents.append(agent);return agent
    def env_factory(*args,**kwargs):
        env=env_type(*args,**kwargs);envs.append(env);return env
    monkeypatch.setattr(train,'DQNAgent',agent_factory);monkeypatch.setattr(train,'SnakeEnv',env_factory)
    progress=[]
    for name,steps in [('plain',()),('snapshots',(24,48,72,96))]:
        seed_cpu(config.seed)
        summary=train.run_training(config,tmp_path/name,make_charts=False,snapshot_steps=steps)
        assert summary['environment_steps_this_run']==96
        progress.append((random.getstate(),torch.get_rng_state().clone()))
    a,b=agents
    assert_equal(a.online.state_dict(),b.online.state_dict())
    assert_equal(a.target.state_dict(),b.target.state_dict())
    assert_equal(a.optimiser.state_dict(),b.optimiser.state_dict())
    assert list(a.replay._memory)==list(b.replay._memory)
    assert a.replay._rng.getstate()==b.replay._rng.getstate()
    assert a._rng.getstate()==b._rng.getstate() and a.episodes_started==b.episodes_started
    assert_equal(progress[0],progress[1])
    assert envs[0].game.snake==envs[1].game.snake and envs[0].game.food==envs[1].game.food
    assert envs[0].game._rng.getstate()==envs[1].game._rng.getstate()
    for filename in ('episodes.csv','updates.csv'):
        assert (tmp_path/'plain'/filename).read_bytes()==(tmp_path/'snapshots'/filename).read_bytes()
    for step in (24,48,72,96):
        agent=DQNAgent.load(tmp_path/f'snapshots/checkpoints/step-{step:06d}.pt')
        assert agent.environment_steps==step and agent.optimiser_steps==step-7
        assert agent.config.max_environment_steps==96
    final=DQNAgent.load(tmp_path/'snapshots/checkpoints/step-000096.pt')
    assert_equal(final.online.state_dict(),b.online.state_dict())


def test_mid_episode_snapshot_does_not_create_boundary(tmp_path):
    config=replace(duration.config_for(duration.load_protocol(),177),episodes=2,max_environment_steps=2,
                   learning_starts=1,batch_size=1)
    seed_cpu(config.seed)
    summary=train.run_training(config,tmp_path/'run',make_charts=False,snapshot_steps=(1,2))
    assert not summary['snapshots']['1']['episode_complete']
    assert summary['snapshots']['1']['episode_length']==1
    assert summary['snapshots']['2']['episode_seed']==summary['snapshots']['1']['episode_seed']
    assert summary['snapshots']['2']['episode_length']==2


@pytest.mark.parametrize('steps',[(0,),(True,),(101,),(32,32),(64,32)])
def test_invalid_snapshot_steps_rejected(tmp_path,steps):
    config=replace(duration.config_for(duration.load_protocol(),177),episodes=100,max_environment_steps=100)
    with pytest.raises(ValueError,match='snapshot_steps'):
        train.run_training(config,tmp_path/'bad',make_charts=False,snapshot_steps=steps)
    assert not (tmp_path/'bad').exists()


def test_protocol_unchanged_algorithm_and_fresh_seed_blocks():
    p=duration.load_protocol()
    assert p['primary_comparison']==[40000,100000] and p['snapshot_steps']==[40000,60000,80000,100000]
    for i,seed in enumerate(duration.SEEDS):
        config=duration.config_for(p,seed)
        assert config.observation_mode=='food13' and config.input_dim==13
        assert config.episodes==config.max_environment_steps==100000
        assert config.training_seed_start==1100000+i*100000
        assert config.evaluation_seed_start==1500000
        assert config.epsilon_decay_steps==50000 and config.learning_starts==1000
    with pytest.raises(ValueError):duration.config_for(p,7)


def test_protocol_and_source_freeze_preserve_old_outputs(tmp_path):
    p=duration.load_protocol();output,meta=duration.prepare(tmp_path/'new',p)
    assert 'snake_ai/experiments/training_duration.py' in meta['source_sha256']
    assert duration.prepare(output,p)==(output,meta)
    p['snapshot_steps']=[40000,100000]
    path=tmp_path/'changed.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):duration.load_protocol(path)
    with pytest.raises(ValueError):duration.prepare(output,p)
    with pytest.raises(ValueError):duration.prepare(duration.DEFAULT_OUTPUT.parent/'milestone5f_body_awareness',p)


def test_evaluation_uses_snapshot_encoder_and_preserves_weights(tmp_path):
    p=duration.load_protocol();config=replace(duration.config_for(p,211,smoke=True),
                                            evaluation_episodes=2,environment=EnvConfig(max_steps=3))
    seed_cpu(config.seed)
    summary=train.run_training(config,tmp_path/'run',make_charts=False,snapshot_steps=(128,256,384,512))
    duration.verify_training(summary,config,(128,256,384,512))
    for step in (128,256,384,512):
        path=tmp_path/f'run/checkpoints/step-{step:06d}.pt';before=digest(path)
        agent=DQNAgent.load(path);rows,metrics=evaluation_rows(agent,f'{step}',211)
        assert metrics['weights_unchanged'] and metrics['input_dim']==13
        assert [r['evaluation_seed'] for r in rows]==[1600000,1600001] and digest(path)==before


def test_successful_trajectory_never_retrained_and_partial_budget_rejected(tmp_path,monkeypatch):
    p=duration.load_protocol();output,meta=duration.prepare(tmp_path/'new',p)
    write_json(output/'validation.json',{'passed':True})
    case=output/'runs/seed177';case.mkdir(parents=True);(case/'data.csv').write_text('keep')
    results=[{'training_seed':177,'checkpoint_steps':40000,'average_score':2.}]
    write_json(case/'complete.json',{'protocol_sha256':meta['protocol_sha256'],'results':results,
                                     'artifact_hashes':{'data.csv':digest(case/'data.csv')}})
    def forbidden(*a,**k):raise AssertionError('Do not repeat successful trajectories')
    monkeypatch.setattr(duration,'run_training',forbidden)
    assert duration.run_one(p,output,177)==results
    (case/'data.csv').write_text('changed')
    with pytest.raises(ValueError):duration.run_one(p,output,177)
    with pytest.raises(RuntimeError):duration.verify_training({'status':'interrupted'},duration.config_for(p,188),duration.STEPS)


def synthetic_results():
    return [{'training_seed':seed,'checkpoint_steps':step,'average_score':float(i*10+step//10000),
        'median_score':2.,'minimum_score':0,'maximum_score':10,'average_episode_length':30.,
        'optimiser_updates':step-999,'epsilon':.24 if step==40000 else .05,
        'completed_training_episodes':500,'training_duration_seconds':60.,
        'collision_rate':1.,'truncation_rate':0.}
        for i,seed in enumerate(duration.SEEDS) for step in duration.STEPS]


def test_within_run_primary_comparison_uses_all_trajectories_and_fixed_endpoint():
    rows=synthetic_results();comparison=duration.compare_results(rows)
    assert comparison['mean_primary_difference']==6.
    assert comparison['checkpoints']['40000']['sd_across_training_seed_means']==10.
    assert comparison['primary_100k_minus_40k']=={'177':6.,'188':6.,'199':6.}
    # A spectacular intermediate value cannot replace the final100k endpoint.
    rows[1]['average_score']=999.
    assert duration.compare_results(rows)['mean_primary_difference']==6.
    with pytest.raises(ValueError):duration.compare_results(rows[:-1])


def test_duration_report_and_charts_from_measurements(tmp_path):
    p=duration.load_protocol();rows=synthetic_results();comparison=duration.compare_results(rows)
    duration.create_charts(tmp_path,rows);duration.write_report(tmp_path,p,rows,comparison)
    assert {x.name for x in tmp_path.glob('*.png')}=={'evaluation_by_step.png','primary_comparison.png'}
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert 'correlated' in text and '1500000-1500099' in text
    with pytest.raises(FileExistsError):duration.write_report(tmp_path,p,rows,comparison)
