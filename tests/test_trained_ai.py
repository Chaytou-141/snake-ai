"""Playback tests use temporary checkpoints and an invisible SDL display."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from snake_ai.agents.dqn_agent import DQNAgent
from snake_ai.demos import trained_ai
from snake_ai.game import Direction, GameStatus
from snake_ai.rl import EnvConfig, RewardConfig
from snake_ai.training.config import TrainingConfig


@pytest.fixture
def checkpoint_factory(tmp_path):
    def create(*, environment=None, rewards=None, filename="model.pt"):
        config = replace(TrainingConfig(), environment=environment or EnvConfig(),
                         rewards=rewards or RewardConfig())
        agent = DQNAgent(config)
        with torch.no_grad():
            agent.online.output_layer.weight.zero_()
            agent.online.output_layer.bias.copy_(torch.tensor([1., 2., 3.]))
        path = tmp_path / filename
        agent.save(path)
        return path
    return create


@pytest.fixture
def session(checkpoint_factory):
    return trained_ai.PlaybackSession(checkpoint_factory(), seed=42)


@pytest.fixture
def pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    yield pygame
    pygame.quit()


def test_checkpoint_load_architecture_cpu_and_environment_config(checkpoint_factory):
    env_config = EnvConfig(width=8, height=7, max_steps=123)
    rewards = RewardConfig(food=3, collision=-2, movement=0.5, win=4)
    path = checkpoint_factory(environment=env_config, rewards=rewards)
    playback = trained_ai.PlaybackSession(path)
    assert playback.env.config == env_config
    assert playback.env.rewards == rewards
    assert playback.model.input_layer.in_features == 11
    assert playback.model.input_layer.out_features == playback.model.hidden_layer.out_features == 128
    assert playback.model.output_layer.out_features == 3
    assert sum(p.numel() for p in playback.model.parameters()) == 18435
    assert all(p.device.type == "cpu" and not p.requires_grad for p in playback.model.parameters())
    assert not playback.model.training
    assert len(playback.observation) == 11
    assert playback.observation == playback.env.observation()


def test_missing_and_mismatched_checkpoints_fail(tmp_path, checkpoint_factory):
    with pytest.raises(FileNotFoundError):
        trained_ai.PlaybackSession(tmp_path / "absent.pt")
    path = checkpoint_factory()
    payload = torch.load(path, weights_only=True)
    payload["online"]["input_layer.weight"] = torch.zeros(128, 12)
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(RuntimeError, match="size mismatch"):
        trained_ai.PlaybackSession(invalid)


def test_argmax_and_inference_mode_without_training_or_mutation(session, monkeypatch):
    weights = [parameter.clone() for parameter in session.model.parameters()]
    counters = (session.agent.environment_steps, session.agent.optimiser_steps, session.agent.epsilon)
    rng = session.agent._rng.getstate()
    calls = []
    def record(module, inputs, output):
        calls.append((torch.is_inference_mode_enabled(), inputs[0].shape, output.requires_grad))
    hook = session.model.register_forward_hook(record)
    def forbidden(*args, **kwargs):
        raise AssertionError("Playback must never train, store experience, or save weights")
    for method in ("remember", "train_step", "save"):
        monkeypatch.setattr(session.agent, method, forbidden)
    for _ in range(3):
        session.step()
        assert session.selected_action == 2
        assert session.q_values == (1., 2., 3.)
    hook.remove()
    assert len(calls) == 3
    assert all(inference and shape == (11,) and not gradients for inference, shape, gradients in calls)
    assert all(torch.equal(a, b) for a, b in zip(weights, session.model.parameters()))
    assert all(parameter.grad is None for parameter in session.model.parameters())
    assert counters == (session.agent.environment_steps, session.agent.optimiser_steps, session.agent.epsilon)
    assert rng == session.agent._rng.getstate()
    assert len(session.agent.replay) == 0


@pytest.mark.parametrize("heading, expected, head", [
    (Direction.UP, Direction.RIGHT, (11, 10)),
    (Direction.RIGHT, Direction.DOWN, (10, 11)),
    (Direction.DOWN, Direction.LEFT, (9, 10)),
    (Direction.LEFT, Direction.UP, (10, 9)),
])
def test_greedy_action_uses_existing_environment_mapping(session, heading, expected, head):
    session.env.game.snake = [(10, 10)]
    session.env.game.direction = heading
    session.env.game.food = (0, 0)
    with torch.no_grad():
        session.model.output_layer.bias.copy_(torch.tensor([-1., 7., 2.]))
    session.observation = session.env.observation()
    session._refresh_decision()
    assert session.selected_action == 1
    session.step()
    assert session.last_action == 1
    assert session.env.game.direction is expected
    assert session.env.game.snake[0] == head


def test_current_q_values_describe_the_next_move(session):
    session.step()
    with torch.inference_mode():
        expected = session.model(torch.tensor(list(session.observation), dtype=torch.float32))
    assert session.q_values == tuple(expected.tolist())
    assert session.selected_action == int(expected.argmax().item())


def test_rewards_and_scores_come_from_environment(session):
    session.env.game.food = (10, 9)  # The fixture's greedy left turn points up.
    session.observation = session.env.observation()
    session._refresh_decision()
    session.step()
    assert session.info["score"] == session.env.game.score == 1
    assert session.last_reward == session.total_reward == 10.0
    assert session.info["steps"] == 1


def test_retry_and_next_episode_seed_behaviour(session):
    initial_obs = session.observation[:]
    initial_food = session.env.game.food
    session.step()
    session.paused = True
    session.reset()
    assert session.episode == 1 and session.seed == 42
    assert session.observation == initial_obs
    assert session.env.game.food == initial_food
    assert session.info["steps"] == 0
    assert session.info["score"] == session.total_reward == session.last_reward == 0
    assert not session.paused and not session.done
    session.reset(next_episode=True)
    assert session.episode == 2 and session.seed == 43
    from snake_ai.rl import SnakeEnv
    reference = SnakeEnv(session.env.config, session.env.rewards)
    assert session.observation == reference.reset(seed=43)[0]


def test_pause_blocks_environment_moves(session):
    session.paused = True
    state = (session.env.game.snake.copy(), session.info.copy(), session.q_values)
    assert session.step() is False
    assert state == (session.env.game.snake, session.info, session.q_values)


def test_collision_stops_playback_and_reset_recovers(session):
    session.env.game.snake = [(0, 0)]
    session.env.game.direction = Direction.RIGHT
    session.observation = session.env.observation()
    session._refresh_decision()  # Relative left points UP, out of the board.
    before = session.q_values
    session.step()
    assert session.terminated and not session.truncated and session.done
    assert session.info["end_reason"] == "collision"
    assert session.last_reward == -10.0
    assert session.q_values == before  # Retained as clearly labelled final-move values.
    assert session.step() is False and session.info["steps"] == 1
    session.reset()
    assert not session.done and session.info["steps"] == 0


def test_truncation_stops_playback_with_engine_still_running(checkpoint_factory):
    playback = trained_ai.PlaybackSession(checkpoint_factory(environment=EnvConfig(max_steps=1)))
    playback.step()
    assert playback.truncated and not playback.terminated
    assert playback.env.game.status is GameStatus.RUNNING
    assert playback.info["end_reason"] == "max_steps"
    assert playback.last_reward == 0.0
    assert playback.step() is False
    playback.reset(next_episode=True)
    assert not playback.done and playback.episode == 2


def test_win_stops_playback(checkpoint_factory):
    playback = trained_ai.PlaybackSession(checkpoint_factory(environment=EnvConfig(width=2, height=2)))
    playback.env.game.snake = [(0, 0), (1, 0), (1, 1)]
    playback.env.game.direction = Direction.LEFT
    playback.env.game.food = (0, 1)
    playback.observation = playback.env.observation()
    playback._refresh_decision()  # Left from LEFT points DOWN to the last cell.
    playback.step()
    assert playback.terminated and playback.info["end_reason"] == "win"
    assert playback.last_reward == 10.0 and playback.info["score"] == 1


def test_speed_changes_clamp_without_affecting_game(session):
    before = (session.observation[:], session.info.copy(), session.q_values)
    session.change_speed(1)
    assert session.speed == 11
    session.change_speed(-1)
    assert session.speed == 10
    session.change_speed(1000)
    assert session.speed == 60
    session.change_speed(-1000)
    assert session.speed == 1
    assert before == (session.observation, session.info, session.q_values)


@pytest.mark.parametrize("speed", [0, 61, float("nan"), float("inf"), True])
def test_invalid_initial_speed(checkpoint_factory, speed):
    with pytest.raises(ValueError, match="speed"):
        trained_ai.PlaybackSession(checkpoint_factory(), speed=speed)


@pytest.mark.parametrize("status", ["running", "paused", "terminated", "truncated"])
def test_headless_drawing_all_playback_states(session, pygame, status):
    pygame.init()
    screen = pygame.display.set_mode(trained_ai.window_size(session))
    if status == "paused":
        session.paused = True
    elif status == "terminated":
        session.terminated = True
        session.info["end_reason"] = "collision"
    elif status == "truncated":
        session.truncated = True
        session.info["end_reason"] = "starvation"
    fonts = {"title": pygame.font.Font(None, 38), "body": pygame.font.Font(None, 25),
             "small": pygame.font.Font(None, 21)}
    trained_ai.draw(screen, session, fonts)
    pygame.display.flip()
    assert screen.get_size() == (940, 700)


@pytest.mark.parametrize("exit_method", ["escape", "close"])
def test_headless_controls_do_not_steer_and_exit_cleanly(session, pygame, monkeypatch, exit_method):
    key = lambda value: pygame.event.Event(pygame.KEYDOWN, key=value)
    exit_event = key(pygame.K_ESCAPE) if exit_method == "escape" else pygame.event.Event(pygame.QUIT)
    batches = iter([
        [key(pygame.K_SPACE)], [], [key(pygame.K_SPACE)], [],
        [key(pygame.K_EQUALS)], [key(pygame.K_KP_MINUS)], [key(pygame.K_UP)],
        [key(pygame.K_r)], [key(pygame.K_n)], [exit_event],
    ])
    observed = []
    original_draw = trained_ai.draw
    def record(screen, playback, fonts):
        observed.append((playback.info["steps"], playback.paused, playback.speed,
                         playback.episode, playback.env.game.direction))
        original_draw(screen, playback, fonts)
    monkeypatch.setattr(pygame.event, "get", lambda: next(batches))
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=lambda fps: 100))
    monkeypatch.setattr(trained_ai, "draw", record)
    trained_ai.run(session)
    assert [row[0] for row in observed] == [0, 0, 0, 1, 1, 1, 2, 0, 0]
    assert [row[1] for row in observed[:4]] == [True, True, False, False]
    assert observed[4][2] == 11 and observed[5][2] == 10
    assert observed[6][4] is Direction.LEFT  # Network left turn, ignoring UP key.
    assert observed[7][3] == 1 and observed[8][3] == 2
    assert not pygame.get_init()


def test_nonfinite_model_outputs_are_rejected(session):
    with torch.no_grad():
        session.model.output_layer.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite Q-values"):
        session._refresh_decision()
