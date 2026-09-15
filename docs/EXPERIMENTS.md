# Exploratory experiments

These measurements came from development runs on one NVIDIA DGX Spark. They were used to select a demo checkpoint. They are not a controlled architecture comparison.

## Graph

```text
MaleCNS release:     v1.0
traced neurons:      165,122
retained edges:      10,511,038
minimum weight:      3 synapses
processed graph:     approximately 202 MB
```

## Search curriculum

The fixed-topology/deep policy reached approximately 0.48 rolling survival against the initial search-16 setup. Factorized plasticity increased the observed rolling value to approximately 0.53 in a subsequent run.

Against search-32, the strongest observed rolling 20-update win rate was approximately 0.510; the final value was approximately 0.495. This is effectively parity within the noise of a single run.

The selected demo checkpoint is:

```text
durak-fast-plastic-r32.best.pt
```

One evaluation of the final search-32 run reported:

```text
counter opponent win rate: 0.6846
seat 0:                    0.6797
seat 1:                    0.6895
```

## Negative hard-teacher result

A hard-label DAgger-style experiment used search-32 actions on 12.5% of agent states with teacher weight 0.1. Teacher loss fell from about 3.20 to 1.14, but agreement stayed near 0.51, normalized policy entropy increased, rolling search win rate ended near 0.458, and counter win rate fell to 0.570.

That checkpoint was rejected. The experiment suggests that stochastic determinization actions should not be treated as single deterministic ground-truth labels. The public trainer therefore defaults to `--teacher-weight 0`.

## Resource observations

The factorized search-32 run used approximately 83 GB at batch size 256 and took roughly 36–43 seconds per update. Memory is dominated by full-connectome activations retained across the 128-decision rollout, not by the stored graph itself.

## Required before scientific claims

- Multiple independent training seeds.
- Paired held-out deal seeds.
- Exact MaleCNS versus degree-preserving rewires and weight/sign controls.
- Parameter- and resource-frontier GRU/LSTM/RNN/MLP baselines.
- Frozen opponent pools not seen during training.
- Confidence intervals, complete logs, configuration hashes, and code revision.
