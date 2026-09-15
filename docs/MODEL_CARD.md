# Model card: FlyDurak demo policy

## Intended use

Interactive demonstrations, reinforcement-learning experiments, sparse recurrent-network research, and reproducibility work around connectome-constrained controllers.

## Not intended for

Claims about animal cognition, biological plausibility, optimal Durak play, gambling, or evidence that MaleCNS topology outperforms conventional recurrent networks.

## Architecture

The release checkpoint is expected to use the processed MaleCNS graph, deep residual decision GRU, factorized source/target plasticity, actor head, and critic head. The graph file is required at inference because sparse indices and base weights are not persisted in the checkpoint.

## Selected checkpoint

```text
durak-fast-plastic-r32.best.pt
```

The file is not stored in Git because it is approximately 374 MB. It is published as a GitHub Release asset and downloaded automatically by `flydurak` when absent.

```text
SHA-256: e9b0388e8aea208ee509f67a04974de563e27ac27090808ebe62327653480285
Download: https://github.com/KremlevLev/flydurak/releases/download/v0.1.0/durak-fast-plastic-r32.best.pt
```

## Limitations

- Trained and evaluated primarily against synthetic policies from the same game engine.
- Search evaluation is stochastic and current reported numbers are single-run estimates.
- Human-game evidence is anecdotal.
- The deep GRU hybrid prevents attributing behavior solely to connectome topology.
- The game and observation model may contain implementation assumptions that differ from regional Durak rules.
- No authentication or abuse protection is included in the browser demo.

## Data

The project downloads MaleCNS flat-connectome tables separately. No source dataset is included in the checkpoint repository. Users are responsible for observing the dataset's attribution and license terms.
