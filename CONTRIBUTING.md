# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest -q tests
```

## Pull requests

- Keep generated datasets, checkpoints, logs, and virtual environments out of Git.
- Add deterministic tests for changes to game rules, legal actions, observations, graph transforms, or checkpoint loading.
- Report the Python, PyTorch, CUDA, GPU, seed, graph threshold, batch size, and opponent configuration with performance claims.
- Do not describe a single training run as evidence that biological topology outperforms a baseline.
- Preserve hidden information: an opponent may use its own hand and public state, but not the other player's cards.

## Reproducible experiments

Commit configuration and analysis code. Publish large checkpoints as release assets or in a model registry, together with hashes and a model card. Raw logs may be attached to a release or archived separately.
