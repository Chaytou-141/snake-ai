"""Fixed demonstration choices: historical original plus first declared seed/final checkpoint."""

from dataclasses import dataclass
from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DemoModel:
    key: str
    label: str
    observation_mode: str
    algorithm: str
    checkpoint: str
    criterion: str


MODELS = (
    DemoModel('original11','Original 11 / seed 7','original11','dqn',
        'results/milestone4_initial/checkpoints/final.pt','Historical original seed 7 final checkpoint.'),
    DemoModel('food13','Food13 / seed 44','food13','dqn',
        'results/milestone5e_food_distance/runs/B_seed44/training/checkpoints/final.pt',
        'First declared 5E training seed; final checkpoint.'),
    DemoModel('geometry14','Geometry14 / seed 11','distance14','dqn',
        'results/milestone5d_feature_comparison/runs/B_seed11/training/checkpoints/final.pt',
        'First declared 5D training seed; final checkpoint.'),
    DemoModel('body16','Body16 / seed 111','body16','dqn',
        'results/milestone5f_body_awareness/runs/B_seed111/training/checkpoints/final.pt',
        'First declared 5F training seed; final checkpoint.'),
    DemoModel('double_dqn','Double DQN / seed 244','food13','double_dqn',
        'results/milestone5h_double_dqn/runs/B_seed244/training/checkpoints/final.pt',
        'First declared 5H training seed; final checkpoint.'),
)


def discover_models(root: Path = PROJECT_ROOT, candidates=MODELS):
    """Offer only present, loadable, compatible files. Never train a replacement."""
    available,notices = [],[]
    try:
        from snake_ai.agents.dqn_agent import DQNAgent
    except ImportError:
        return [],['AI playback requires the ML extra: pip install -e ".[training]"']
    for entry in candidates:
        path = Path(root)/entry.checkpoint
        if not path.is_file():
            notices.append(f'{entry.key}: checkpoint missing locally')
            continue
        try:
            agent = DQNAgent.load(path)
            if (agent.config.observation_mode,agent.config.algorithm)!=(entry.observation_mode,entry.algorithm):
                raise ValueError('Checkpoint configuration differs from catalog')
            if agent.online.input_layer.in_features != agent.config.input_dim:
                raise ValueError('Checkpoint input size differs from encoder')
            if not all(p.isfinite().all().item() for p in agent.online.parameters()):
                raise ValueError('Checkpoint contains nonfinite inference weights')
            available.append(entry)
        except (OSError,ValueError,RuntimeError,KeyError,TypeError,EOFError,pickle.UnpicklingError) as error:
            notices.append(f'{entry.key}: incompatible checkpoint ({type(error).__name__})')
    return available,notices


class ModelController:
    def __init__(self, models, *, root: Path = PROJECT_ROOT):
        self.models = tuple(models)
        self.root = Path(root)
        if not self.models:
            raise ValueError('No compatible local model checkpoints are available')

    def open(self, key: str, *, seed=800000, speed=10.):
        from snake_ai.demos.trained_ai import PlaybackSession
        entry = next((m for m in self.models if m.key==key),None)
        if entry is None:
            raise ValueError(f'Unavailable model: {key}')
        session = PlaybackSession(self.root/entry.checkpoint,seed=seed,speed=speed,model_name=entry.label)
        if (session.agent.config.observation_mode,session.agent.config.algorithm)!=(entry.observation_mode,entry.algorithm):
            raise ValueError('Checkpoint configuration changed since model discovery')
        session.demo_key = key
        session.model_switching = True
        return session

    def switch(self, session, delta: int):
        index = next(i for i,m in enumerate(self.models) if m.key==session.demo_key)
        entry = self.models[(index+delta)%len(self.models)]
        # Construct transactionally: a missing/invalid replacement never changes the old session.
        replacement = self.open(entry.key,seed=session.base_seed,speed=session.speed)
        replacement.episode = session.episode
        replacement.reset()
        replacement.paused = True
        replacement.playback_message = 'Model switched; same seed. Space: resume.'
        return replacement
