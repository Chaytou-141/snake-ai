"""Target math, gradient isolation, backward compatibility and matched protocols."""

from dataclasses import replace
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.experiments import feature_comparison as experiment
from snake_ai.rl import EnvConfig, SnakeEnv
from snake_ai.training.config import TrainingConfig
from snake_ai.training.evaluate import evaluate_agent
from snake_ai.training.replay_buffer import ReplayBatch
from snake_ai.training.train import run_training

PROTOCOL=Path('configs/double_dqn_v1.json')


@pytest.fixture(autouse=True)
def restore_state():
    state=random.getstate();threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):yield
    random.setstate(state);torch.set_num_threads(threads);torch.use_deterministic_algorithms(det)


class FixedQ(nn.Module):
    def __init__(self,values):
        super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float32))
    def forward(self,states):return self.values.expand(len(states),-1)


@pytest.mark.parametrize('algorithm,expected',[('dqn',[6.,2.,8.]),('double_dqn',[2.,2.,4.])])
def test_targets_action_selection_terminal_truncation_and_gradient_isolation(algorithm,expected):
    agent=DQNAgent(replace(TrainingConfig(),algorithm=algorithm,gamma=.5))
    agent.online=FixedQ([3.,9.,1.]);agent.target=FixedQ([10.,2.,0.])
    next_states=torch.zeros((3,11),requires_grad=True)
    rewards=torch.tensor([1.,2.,3.],requires_grad=True)
    batch=ReplayBatch(torch.zeros((3,11)),torch.zeros(3,dtype=torch.long),rewards,next_states,
        torch.tensor([False,True,False]),torch.tensor([False,False,True]))
    targets=agent.compute_targets(batch)
    assert targets.tolist()==expected and not targets.requires_grad and targets.grad_fn is None
    assert rewards.grad is None and next_states.grad is None
    assert agent.online.values.grad is None and agent.target.values.grad is None
    # Regression gradients belong only to Q(state,action), never to the target calculation.
    agent.online(batch.states)[:,0].sub(targets).square().mean().backward()
    assert agent.online.values.grad is not None and agent.target.values.grad is None
    assert next_states.grad is None and rewards.grad is None


def test_standard_target_does_not_consult_online_action_selection():
    class Forbidden(nn.Module):
        def forward(self,x):raise AssertionError('Standard target must only use target network')
    agent=DQNAgent();agent.online=Forbidden();agent.target=FixedQ([2.,7.,3.])
    batch=ReplayBatch(torch.zeros((1,11)),torch.tensor([0]),torch.tensor([1.]),torch.zeros((1,11)),
                     torch.tensor([False]),torch.tensor([False]))
    assert agent.compute_targets(batch).item()==pytest.approx(1.+.99*7.)


@pytest.mark.parametrize('algorithm',['dqn','double_dqn'])
def test_algorithm_checkpoint_roundtrip_and_seeded_reproducibility(tmp_path,algorithm):
    config=replace(TrainingConfig(),algorithm=algorithm,observation_mode='food13',seed=288,
                   learning_starts=2,batch_size=2)
    a,b=DQNAgent(config),DQNAgent(config);obs,_=SnakeEnv(observation_mode='food13').reset(seed=600000)
    assert torch.equal(a.predict(obs),b.predict(obs))
    for agent in (a,b):
        for _ in range(2):agent.remember(obs,0,1.,obs,False,False)
        assert agent.train_step() is not None
    assert all(torch.equal(x,y) for x,y in zip(a.online.parameters(),b.online.parameters()))
    path=tmp_path/'model.pt';a.save(path);before=experiment.digest(path)
    loaded=DQNAgent.load(path)
    assert loaded.config.algorithm==algorithm and torch.equal(a.predict(obs),loaded.predict(obs))
    assert experiment.digest(path)==before
    assert ('algorithm' in config.to_dict())==(algorithm=='double_dqn')


def test_default_standard_schema_and_algorithm_validation():
    assert TrainingConfig().algorithm=='dqn' and 'algorithm' not in TrainingConfig().to_dict()
    assert TrainingConfig.from_dict(TrainingConfig().to_dict()).algorithm=='dqn'
    for invalid in ('unknown',None,True):
        with pytest.raises(ValueError):replace(TrainingConfig(),algorithm=invalid)


def test_existing_original_and_food13_checkpoints_read_only():
    paths=[Path('results/milestone4_initial/checkpoints/final.pt'),
           Path('results/milestone5g_training_duration/runs/seed177/training/checkpoints/final.pt')]
    if not all(p.exists() for p in paths):pytest.skip('Historical checkpoints excluded from Git')
    for path in paths:
        before=experiment.digest(path);agent=DQNAgent.load(path)
        obs,_=SnakeEnv(observation_mode=agent.config.observation_mode).reset(seed=600000)
        assert agent.config.algorithm=='dqn' and torch.isfinite(agent.predict(obs)).all()
        assert experiment.digest(path)==before


def test_paired_algorithm_protocol_and_identical_initial_weights():
    p=experiment.load_protocol(PROTOCOL)
    assert p['training_seeds']==[244,255,266] and p['evaluation_seed_start']==970000
    for seed in p['training_seeds']:
        a,b=(experiment.training_config(p,m,seed) for m in 'AB')
        da,db=a.to_dict(),b.to_dict();assert db.pop('algorithm')=='double_dqn';assert da==db
        assert a.max_environment_steps==a.episodes==100000 and a.input_dim==b.input_dim==13
        aa,bb=DQNAgent(a),DQNAgent(b)
        assert all(torch.equal(x,y) for x,y in zip(aa.online.parameters(),bb.online.parameters()))
        assert aa.replay._rng.getstate()==bb.replay._rng.getstate() and aa._rng.getstate()==bb._rng.getstate()
    assert experiment.training_config(p,'B',288,smoke=True).evaluation_seed_start==980000
    with pytest.raises(ValueError):experiment.training_config(p,'A',211)


def test_recording_next_states_has_no_policy_or_parameter_side_effects():
    p=experiment.load_protocol(PROTOCOL)
    config=replace(experiment.training_config(p,'B',288,smoke=True),evaluation_episodes=2,
                   environment=EnvConfig(max_steps=12))
    agent=DQNAgent(config);before=[p.clone() for p in agent.online.parameters()]
    baseline,base_metrics=evaluate_agent(agent,include_random=False)
    states=[]
    recorded,metrics=evaluate_agent(agent,include_random=False,state_recorder=lambda *args:states.append(args))
    assert recorded==baseline and metrics==base_metrics and states
    assert all(torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    assert all(len(row[2])==13 for row in states)


def test_fixed_capture_schedule_skips_terminal_and_truncated_states():
    p=experiment.load_protocol(PROTOCOL);states,capture=experiment.state_recorder(p,'A',244)
    obs=(0.,)*13
    for seed,move,term,trunc in [(970000,1,False,False),(970000,2,False,False),
        (970020,1,False,False),(970000,10,True,False),(970000,25,False,True),(970019,50,False,False)]:
        capture(seed,move,obs,term,trunc)
    assert len(states)==2 and [r['move'] for r in states]==[1,50]
    assert states[0]['origin_model']=='A' and states[0]['feature_12']==0.


def test_diagnostic_measures_online_max_minus_target_at_online_action():
    agent=DQNAgent();agent.online=FixedQ([3.,9.,1.]);agent.target=FixedQ([10.,2.,0.])
    states=torch.zeros((4,11),requires_grad=True)
    gap,online,target=experiment.q_disagreement(agent,states)
    assert gap.tolist()==[7.]*4 and online.tolist()==[9.]*4 and target.tolist()==[2.]*4
    assert all(not v.requires_grad for v in (gap,online,target))


def test_real_common_state_pool_is_used_unchanged_for_both_algorithms(tmp_path):
    p=experiment.load_protocol(PROTOCOL)
    # The same pooling implementation runs on a smaller synthetic pair without extra episodes.
    small={**p,'run_order':[['A',244],['B',244]]}
    for model in 'AB':
        case=tmp_path/f'runs/{model}_seed244';case.mkdir(parents=True)
        config=experiment.training_config(p,model,244);agent=DQNAgent(config);agent.optimiser_steps=99001
        agent.save(case/'training/checkpoints/final.pt')
        rows=[{'origin_model':model,'training_seed':244,'evaluation_seed':970000,'move':1,
               **{f'feature_{i}':float(model=='B') for i in range(13)}}]
        experiment.write_csv(case/'diagnostic_states.csv',rows)
    result=experiment.run_q_diagnostic(tmp_path,small)
    assert result['common_states']==2 and len(result['per_run'])==2
    assert all(r['states']==2 and r['target_update_lag']==1 for r in result['per_run'])
    assert experiment.run_q_diagnostic(tmp_path,small)==result
    (tmp_path/'q_disagreement.csv').write_text('changed')
    with pytest.raises(ValueError):experiment.run_q_diagnostic(tmp_path,small)


def test_protocol_freeze_and_historical_output_protection(tmp_path):
    p=experiment.load_protocol(PROTOCOL);output,_=experiment.prepare_output(tmp_path/'new',p)
    changed=json.loads(json.dumps(p));changed['algorithms']['B']='dqn'
    path=tmp_path/'changed.json';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):experiment.load_protocol(path)
    with pytest.raises(ValueError):experiment.prepare_output(output,changed)
    with pytest.raises(ValueError):experiment.prepare_output(Path('results/milestone5g_training_duration'),p)


@pytest.mark.parametrize('algorithm',['dqn','double_dqn'])
def test_training_and_greedy_evaluation_match_algorithm_encoder(tmp_path,algorithm):
    p=experiment.load_protocol(PROTOCOL)
    config=replace(experiment.training_config(p,'A' if algorithm=='dqn' else 'B',244),
        episodes=64,max_environment_steps=64,learning_starts=8,batch_size=4,environment=EnvConfig(max_steps=1),
        evaluation_seed_start=980000,evaluation_episodes=2)
    experiment.seed_cpu(config.seed)
    summary=run_training(config,tmp_path/'run',make_charts=False)
    assert summary['status']=='environment_step_cap' and summary['environment_steps_this_run']==64
    assert summary['optimiser_updates_this_run']==57 and summary['weights_changed']
    agent=DQNAgent.load(tmp_path/'run/checkpoints/final.pt')
    before=experiment.digest(tmp_path/'run/checkpoints/final.pt')
    rows,metrics=experiment.evaluation_rows(agent,'A',244)
    assert len(rows)==2 and metrics['weights_unchanged'] and agent.config.algorithm==algorithm
    assert experiment.digest(tmp_path/'run/checkpoints/final.pt')==before


def test_algorithm_report_chart_and_incomplete_run_guard(tmp_path,monkeypatch):
    p=experiment.load_protocol(PROTOCOL)
    rows=[{'model':m,'training_seed':s,'average_score':float(s%10)+(1 if m=='B' else 0),
        'completed_training_episodes':800,'training_environment_steps':100000,'optimiser_updates':99001,
        'training_duration_seconds':220.,'median_score':4.,'minimum_score':0,'maximum_score':10,
        'average_episode_length':70.,'collision_rate':1.,'truncation_rate':0.,'win_rate':0.,
        'mean_loss':.1,'maximum_loss':2.,'last10000_mean_loss':.1,'last10000_loss_sd':.01,
        'gradient_clipped_fraction':.02} for s in (244,255,266) for m in 'AB']
    comparison=experiment.compare_results(rows,p['training_seeds'])
    diagnostic={'common_states':2,'interpretation':'Estimator disagreement, not actual bias.',
        'per_run':[{'model':r['model'],'training_seed':r['training_seed'],'mean_gap':.1,'mean_abs_gap':.2,
                    'sd_gap':.3,'target_update_lag':1} for r in rows]}
    experiment.create_charts(tmp_path,rows,p)
    experiment.write_algorithm_report(tmp_path,rows,comparison,diagnostic,p)
    assert {x.name for x in tmp_path.glob('*.png')}=={'average_score.png'}
    text=(tmp_path/'report.md').read_text(encoding='utf-8')
    assert 'Double DQN' in text and '970000-970099' in text and 'disagreement' in text
    output,_=experiment.prepare_output(tmp_path/'new',p)
    experiment.write_json(output/'validation.json',{'passed':True})
    monkeypatch.setattr(experiment,'run_training',lambda *a,**k:{'status':'environment_step_cap',
        'environment_steps_this_run':99999,'completed_episodes_this_run':800})
    with pytest.raises(RuntimeError,match='Incomplete fixed-step algorithm'):
        experiment.run_one(p,output,'A',244)
