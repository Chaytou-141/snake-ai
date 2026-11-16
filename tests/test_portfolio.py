"""Local model selection, inference controls and saved-data dashboard verification."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from snake_ai import dashboard, demo
from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.demos import trained_ai
from snake_ai.demos.catalog import MODELS, PROJECT_ROOT, ModelController, discover_models
from snake_ai.training.config import TrainingConfig


@pytest.fixture
def local_models(tmp_path):
    for entry in MODELS:
        config=replace(TrainingConfig(),observation_mode=entry.observation_mode,algorithm=entry.algorithm)
        DQNAgent(config).save(tmp_path/entry.checkpoint)
    return tmp_path


@pytest.fixture
def pygame(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    import pygame
    yield pygame
    pygame.quit()


def test_fixed_catalog_loads_all_encoders_and_algorithms(local_models,monkeypatch):
    models,notes=discover_models(local_models)
    assert models==list(MODELS) and not notes
    controller=ModelController(models,root=local_models)
    for entry in models:
        path=local_models/entry.checkpoint;before=hashlib.sha256(path.read_bytes()).hexdigest()
        session=controller.open(entry.key,seed=800000)
        assert session.agent.config.observation_mode==entry.observation_mode
        assert session.agent.config.algorithm==entry.algorithm
        assert len(session.observation)==session.agent.config.input_dim
        state=(session.agent.environment_steps,session.agent.optimiser_steps,session.agent._rng.getstate())
        weights=[p.clone() for p in session.model.parameters()]
        def forbidden(*a,**k):raise AssertionError('Playback must not train or save models')
        for name in ('train_step','remember','save'):monkeypatch.setattr(session.agent,name,forbidden)
        for _ in range(4):session.step()
        assert state==(session.agent.environment_steps,session.agent.optimiser_steps,session.agent._rng.getstate())
        assert all(torch.equal(a,b) for a,b in zip(weights,session.model.parameters()))
        assert not session.model.training and all(not p.requires_grad for p in session.model.parameters())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_missing_and_incompatible_models_are_not_offered(tmp_path,local_models):
    models,notes=discover_models(tmp_path/'missing')
    assert models==[] and len(notes)==5
    with pytest.raises(ValueError,match='No compatible'):ModelController(models,root=tmp_path)
    wrong=local_models/MODELS[1].checkpoint
    DQNAgent().save(local_models/'wrong.pt')
    shutil.copyfile(local_models/'wrong.pt',wrong)
    models,notes=discover_models(local_models)
    assert 'food13' not in [m.key for m in models] and any('incompatible' in n for n in notes)


def test_switching_resets_encoder_preserves_seed_and_keeps_old_session_on_failure(local_models):
    controller=ModelController(MODELS,root=local_models)
    old=controller.open('original11',seed=800000,speed=17)
    old.reset(next_episode=True);old.step()
    seed=old.seed
    new=controller.switch(old,1)
    assert new.demo_key=='food13' and new.seed==seed and new.episode==old.episode
    assert new.speed==17 and new.info['steps']==0 and new.paused
    assert len(new.observation)==13
    reference=controller.open('original11',seed=seed)
    assert new.observation[:11]==reference.observation and new.env.game.food==reference.env.game.food
    (local_models/MODELS[2].checkpoint).unlink()
    before=(new.seed,new.info.copy(),new.q_values)
    with pytest.raises(FileNotFoundError):controller.switch(new,1)
    assert (new.seed,new.info,new.q_values)==before


def test_switch_controls_and_headless_draw(local_models,pygame,monkeypatch):
    controller=ModelController(MODELS,root=local_models)
    session=controller.open('original11')
    key=lambda k:pygame.event.Event(pygame.KEYDOWN,key=k,mod=0)
    batches=iter([[key(pygame.K_TAB)],[key(pygame.K_SPACE)],[key(pygame.K_r)],
                  [key(pygame.K_n)],[key(pygame.K_EQUALS)],[key(pygame.K_MINUS)],[key(pygame.K_ESCAPE)]])
    seen=[];draw=trained_ai.draw
    def record(screen,current,fonts):
        seen.append((current.demo_key,current.paused,current.episode,current.speed))
        draw(screen,current,fonts)
    monkeypatch.setattr(pygame.event,'get',lambda:next(batches))
    monkeypatch.setattr(pygame.time,'Clock',lambda:SimpleNamespace(tick=lambda fps:100))
    monkeypatch.setattr(trained_ai,'draw',record)
    trained_ai.run(session,switch_model=controller.switch)
    assert seen[0][:2]==('food13',True) and seen[1][:2]==('food13',False)
    assert seen[2][2]==1 and seen[3][2]==2
    assert seen[4][3]==11 and seen[5][3]==10 and not pygame.get_init()


@pytest.mark.parametrize('choice',['human','model','exit'])
def test_menu_keyboard_navigation_without_inference(local_models,pygame,monkeypatch,choice):
    key=lambda k:pygame.event.Event(pygame.KEYDOWN,key=k)
    events={'human':[key(pygame.K_h)],'model':[key(pygame.K_DOWN),key(pygame.K_RETURN)],'exit':[key(pygame.K_ESCAPE)]}
    monkeypatch.setattr(pygame.event,'get',lambda:events[choice])
    result=demo.choose(MODELS,[],message='')
    assert result=={'human':'human','model':'original11','exit':None}[choice]
    assert not pygame.get_init()


def test_cli_missing_model_is_clear_error_without_traceback(monkeypatch,capsys):
    monkeypatch.setattr(demo,'discover_models',lambda:([],['missing']))
    monkeypatch.setattr('sys.argv',['demo','--model','food13'])
    with pytest.raises(SystemExit) as error:demo.main()
    assert error.value.code==2 and 'No compatible local model' in capsys.readouterr().err


def test_corrupt_and_nonfinite_checkpoints_are_handled(local_models,monkeypatch,capsys):
    path=local_models/MODELS[0].checkpoint
    path.write_text('This is not a checkpoint')
    models,notes=discover_models(local_models)
    assert 'original11' not in [m.key for m in models] and any('incompatible' in n for n in notes)
    monkeypatch.setattr('sys.argv',['demo','--checkpoint',str(path)])
    with pytest.raises(SystemExit) as error:demo.main()
    assert error.value.code==2 and 'Checkpoint unavailable' in capsys.readouterr().err
    path=local_models/MODELS[1].checkpoint;payload=torch.load(path,weights_only=True)
    payload['online']['output_layer.bias'].fill_(float('nan'));torch.save(payload,path)
    models,notes=discover_models(local_models)
    assert 'food13' not in [m.key for m in models]


@pytest.fixture
def data_project(tmp_path):
    data=dashboard.load_data(PROJECT_ROOT)
    files=list(data['source_sha256'])+[s['chart'] for s in data['studies']]+[
        'results/milestone5g_training_duration/evaluation_by_step.png']
    for relative in files:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/relative,target)
    return tmp_path


def test_dashboard_reproduces_saved_aggregates_and_preserves_results(data_project):
    root=data_project
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'results').rglob('*') if p.is_file()}
    data=dashboard.load_data(root)
    assert data['initial']['random']['average_score']==.14
    assert data['initial']['dqn_greedy']['average_score']==34.81
    assert len(data['studies'])==4 and len(data['duration_rows'])==12
    assert data['duration']['mean_primary_difference']==pytest.approx(.19666666666666663)
    output=dashboard.build_dashboard(root,root/'portfolio/dashboard')
    html=output.read_text(encoding='utf-8')
    assert 'not a universal model ranking' in html and '970000' in html and 'correlated' in html
    saved=json.loads(output.with_name('data.json').read_text())
    assert saved==data
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
    with pytest.raises(ValueError):dashboard.build_dashboard(root,root/'results/danger')


def test_dashboard_rejects_manually_changed_saved_scores(data_project):
    path=data_project/'results/milestone5e_food_distance/comparison_summary.json'
    data=json.loads(path.read_text());data['variants']['A']['mean_of_training_seed_means']+=1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='aggregate'):dashboard.load_data(data_project)


def test_dashboard_missing_results_fail_instead_of_inventing_values(tmp_path):
    with pytest.raises(FileNotFoundError):dashboard.load_data(tmp_path)
