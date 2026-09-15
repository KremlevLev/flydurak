import math

import torch

from flysans.train import TrainConfig, train


def test_training_smoke_has_finite_metrics():
    config = TrainConfig(updates=2, batch_size=4, horizon=12, neurons=24, edges_per_neuron=2, eval_episodes=8)
    _, metrics, _ = train(config, torch.device("cpu"), model_type="mlp")
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["evaluation"]["greedy"]["survival"] <= 1.0
