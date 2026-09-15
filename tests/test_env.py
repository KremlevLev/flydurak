import pytest
import torch

from flysans.env import BoneSweepEnv, EnvConfig


def test_reset_is_deterministic_and_actions_stay_in_bounds():
    env = BoneSweepEnv(EnvConfig(batch_size=4, horizon=10, curriculum_level=4))
    first = env.reset(42).clone()
    second = env.reset(42).clone()
    assert torch.equal(first, second)
    for action in range(5):
        observation, _, _, _ = env.step(torch.full((4,), action))
        assert observation[:, :2].abs().max() <= 1.0


def test_invalid_action_is_rejected():
    env = BoneSweepEnv(EnvConfig(batch_size=2))
    with pytest.raises(ValueError):
        env.step(torch.tensor([0, 5]))


def test_oracle_survives_every_attack():
    for attack in range(len(BoneSweepEnv.attack_names)):
        env = BoneSweepEnv(EnvConfig(batch_size=64, horizon=90, forced_attack=attack))
        env.reset(17)
        survived = None
        for _ in range(env.config.horizon):
            _, _, _, info = env.step(env.oracle_action())
            survived = info["survived"]
        assert float(survived.float().mean()) >= 0.95, BoneSweepEnv.attack_names[attack]


def test_forced_attack_fills_the_batch():
    env = BoneSweepEnv(EnvConfig(batch_size=8, forced_attack=1))
    env.reset(3)
    assert bool((env.attack == 1).all())
