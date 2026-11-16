"""SnakeAI1.0 local launcher: human play and compatible saved-model playback."""

import argparse
import math
from pathlib import Path
import pickle
import webbrowser

from snake_ai.demos.catalog import PROJECT_ROOT, ModelController, discover_models


def choose(models, notices, *, message=''):
    import pygame
    pygame.init()
    try:
        screen = pygame.display.set_mode((940,700))
        pygame.display.set_caption('SnakeAI1.0 / Play and inspect')
        fonts = {'title':pygame.font.SysFont('segoeui',54,bold=True),
                 'body':pygame.font.SysFont('segoeui',23), 'small':pygame.font.SysFont('segoeui',17),
                 'heading':pygame.font.SysFont('segoeui',30,bold=True)}
        clock = pygame.time.Clock()
        selected = 0
        while True:
            screen.fill((14,23,33))
            def text(value,x,y,face='body',color=(233,240,245)):
                screen.blit(fonts[face].render(value,True,color),(x,y))
            text('SnakeAI',32,25,'title')
            text('Play the game. Inspect the agent.',35,103,color=(168,189,204))
            text('Version1.0  /  CPU inference',647,55,'small',(168,189,204))
            rectangles=[pygame.Rect(32,181,300,328)]
            pygame.draw.rect(screen,(25,43,53),rectangles[0],border_radius=12)
            if selected==0:pygame.draw.rect(screen,(139,244,179),rectangles[0],width=2,border_radius=12)
            text('Human play',53,200,'heading')
            text('Arrow keys to steer.',53,249,'small')
            text('R retries. Esc returns.',53,275,'small')
            for i,(x,y) in enumerate(((60,350),(87,350),(114,350),(114,323),(141,323))):
                pygame.draw.rect(screen,(139,244,179) if i==4 else (56,185,126),(x,y,23,23),border_radius=5)
            pygame.draw.rect(screen,(255,113,107),(230,323,23,23),border_radius=5)
            text('H  Play with keyboard',53,449,'small',(139,244,179))
            text('Watch a trained agent',374,178,'heading')
            text('Only compatible local checkpoints are shown.',375,220,'small',(168,189,204))
            for i,entry in enumerate(models):
                rect=pygame.Rect(373,260+i*48,535,43);rectangles.append(rect)
                pygame.draw.rect(screen,(36,71,57) if selected==i+1 else (25,37,49),rect,border_radius=6)
                if selected==i+1:pygame.draw.rect(screen,(139,244,179),rect,width=1,border_radius=6)
                text(entry.label,386,rect.y+4,'body')
                text(entry.algorithm,765,rect.y+11,'small',(168,189,204))
            if not models:
                text('No local model weights found.',375,273,'body')
                text('Human play is ready. See README for checkpoint paths.',375,315,'small',(168,189,204))
            text('Selection uses fixed seed/final checkpoints, never a score ranking.',375,521,'small',(168,189,204))
            if notices:text(f'{len(notices)} unavailable model(s). --list-models shows details.',35,565,'small',(230,183,119))
            if message:
                text(message[:103],35,594,'small',(230,183,119))
            text('Up/Down: select   Enter: launch   H: human   D: dashboard   Esc: exit',35,644,'small',(168,189,204))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type==pygame.QUIT:return None
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:return None
                    if event.key==pygame.K_h:return 'human'
                    if event.key==pygame.K_d:
                        path=PROJECT_ROOT/'portfolio/dashboard/index.html'
                        if path.exists():webbrowser.open(path.as_uri())
                        else:message='Generate the dashboard: python -m snake_ai.dashboard'
                    if event.key in (pygame.K_UP,pygame.K_DOWN):
                        selected=(selected+(-1 if event.key==pygame.K_UP else 1))%(len(models)+1)
                    if event.key==pygame.K_RETURN:return 'human' if selected==0 else models[selected-1].key
                if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    for i,rect in enumerate(rectangles):
                        if rect.collidepoint(event.pos):return 'human' if i==0 else models[i-1].key
            clock.tick(30)
    finally:
        pygame.quit()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--human',action='store_true')
    group.add_argument('--model',help='Fixed catalog key; --list-models shows available choices')
    group.add_argument('--checkpoint',type=Path,help='Explicit compatible local checkpoint')
    parser.add_argument('--list-models',action='store_true')
    parser.add_argument('--seed',type=int,default=800000)
    parser.add_argument('--speed',type=float,default=10.)
    args=parser.parse_args()
    if args.seed<0 or not math.isfinite(args.speed) or not 1<=args.speed<=60:
        parser.error('Use a nonnegative seed and speed between1 and60')
    if args.human:
        from snake_ai.game.engine import SnakeGame
        from snake_ai.game.renderer import run
        run(SnakeGame(seed=args.seed),moves_per_second=args.speed)
        return
    if args.checkpoint is not None:
        from snake_ai.demos.trained_ai import PlaybackSession,run
        try:session=PlaybackSession(args.checkpoint,seed=args.seed,speed=args.speed)
        except (OSError,ValueError,RuntimeError,KeyError,TypeError,EOFError,pickle.UnpicklingError) as error:parser.error(f'Checkpoint unavailable: {error}')
        run(session)
        return
    models,notices=discover_models()
    if args.list_models:
        for m in models:print(f'{m.key:12} {m.observation_mode:12} {m.algorithm:12} {m.checkpoint}')
        for note in notices:print(note)
        if not models:print('No compatible local models. Human play: python -m snake_ai --seed 800000')
        return
    if args.model:
        from snake_ai.demos.trained_ai import run
        try:
            controller=ModelController(models)
            session=controller.open(args.model,seed=args.seed,speed=args.speed)
        except (OSError,ValueError,RuntimeError,KeyError,TypeError,EOFError,pickle.UnpicklingError) as error:parser.error(str(error))
        run(session,switch_model=controller.switch)
        return
    message=''
    while True:
        choice=choose(models,notices,message=message)
        if choice is None:return
        if choice=='human':
            from snake_ai.game.engine import SnakeGame
            from snake_ai.game.renderer import run
            run(SnakeGame(seed=args.seed),moves_per_second=args.speed)
        else:
            from snake_ai.demos.trained_ai import run
            try:
                controller=ModelController(models)
                session=controller.open(choice,seed=args.seed,speed=args.speed)
                run(session,switch_model=controller.switch)
                message=''
            except (OSError,ValueError,RuntimeError,KeyError,TypeError,EOFError,pickle.UnpicklingError):
                message='Model unavailable. Supply a compatible checkpoint; nothing was trained.'
                models,notices=discover_models()


if __name__=='__main__':main()
