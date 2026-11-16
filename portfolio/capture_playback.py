"""Capture genuine local inference frames. No training or experiment evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

from PIL import Image
import pygame
import torch
from snake_ai.demos import trained_ai
from snake_ai.demos.catalog import PROJECT_ROOT,ModelController,discover_models
from snake_ai import demo


def capture(output:Path,model='original11',seed=800000):
    output=output.resolve()
    if output.is_relative_to(PROJECT_ROOT/'results'):raise ValueError('Do not overwrite historical experiments')
    names=['playback.png','playback.gif','capture.json','menu.png']
    if any((output/name).exists() for name in names):raise FileExistsError('Use a new capture directory')
    output.mkdir(parents=True,exist_ok=True)
    models,notices=discover_models();controller=ModelController(models)
    session=controller.open(model,seed=seed,speed=10.)
    before=hashlib.sha256(session.checkpoint.read_bytes()).hexdigest()
    weights=[p.clone() for p in session.model.parameters()]
    progress=(session.agent.environment_steps,session.agent.optimiser_steps,session.agent._rng.getstate())
    pygame.init()
    try:
        screen=pygame.display.set_mode(trained_ai.window_size(session))
        fonts={'title':pygame.font.Font(None,38),'body':pygame.font.Font(None,25),'small':pygame.font.Font(None,21)}
        frames=[];screenshot=False
        for move in range(121):
            if move:session.step()
            if move%2==0 or session.done:
                trained_ai.draw(screen,session,fonts);pygame.display.flip()
                frame=Image.frombytes('RGB',screen.get_size(),pygame.image.tobytes(screen,'RGB'))
                frames.append(frame.resize((752,560)).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
            if move==60 or session.done:
                if not screenshot:pygame.image.save(screen,str(output/'playback.png'));screenshot=True
            if session.done:break
        if not screenshot:pygame.image.save(screen,str(output/'playback.png'))
        frames[0].save(output/'playback.gif',save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=True)
    finally:pygame.quit()
    assert hashlib.sha256(session.checkpoint.read_bytes()).hexdigest()==before
    assert all(torch.equal(a,b) for a,b in zip(weights,session.model.parameters()))
    assert progress==(session.agent.environment_steps,session.agent.optimiser_steps,session.agent._rng.getstate())
    event_get=pygame.event.get
    def exit_after_frame():
        pygame.image.save(pygame.display.get_surface(),str(output/'menu.png'))
        return [pygame.event.Event(pygame.QUIT)]
    pygame.event.get=exit_after_frame
    try:demo.choose(models,notices)
    finally:pygame.event.get=event_get
    metadata={'model':model,'checkpoint':str(session.checkpoint.relative_to(PROJECT_ROOT)),
        'checkpoint_sha256':before,'seed':seed,'frames':len(frames),'frame_duration_ms':200,
        'moves_per_frame':2,'playback_speed':10,'moves':session.info['steps'],'score':session.info['score'],
        'weights_unchanged':True,'interpretation':'Actual Pygame inference capture; first120 moves or episode end. '
            'Every second move rendered at200ms per frame. Demonstration only, not an evaluation result.'}
    (output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'portfolio/assets')
    parser.add_argument('--model',default='original11');parser.add_argument('--seed',type=int,default=800000)
    args=parser.parse_args();capture(args.output,args.model,args.seed)
