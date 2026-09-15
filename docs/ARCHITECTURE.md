# Architecture

## Game

`DurakGame` implements two-player, 36-card podkidnoy Durak. An action is one of 36 physical cards, `TAKE`, or `PASS`. Illegal actions are masked before sampling. The observation contains the acting player's hand, table cards, trump suit, phase, deck and opponent-hand sizes, role, and public discard history. The opponent's hand is not included.

## Connectome conversion

`flysans-prepare` selects traced MaleCNS bodies, filters neuron-to-neuron connections by synapse count, applies `log1p` to counts, infers a coarse transmitter group, assigns inhibitory signs to GABA/glutamate/histamine edges, and normalizes absolute incoming weight per target neuron. Indices are stored in target/source order for PyTorch sparse multiplication.

## Controller

At every decision, the observation is projected into all connectome neurons. The recurrent update is:

```text
source-scaled state
→ fixed sparse MaleCNS propagation
→ target scaling
→ observation injection
→ tanh + learned leak
```

The `deep` configuration adds a 256-unit GRU decision state and residual policy/value heads. It is therefore a hybrid controller; results from it do not isolate an advantage of biological topology.

## Factorized plasticity

With `--plastic-edges`, every source and target neuron has a bounded gain. The effective edge weight is:

```text
w'(target, source) = w × source_gain × target_gain
```

Both gains start at 1.0 and remain between 0.5 and 1.5. This changes the effective strength of every retained edge without differentiating through sparse matrix values, which avoids a dense `165122 × 165122` adjacency gradient.

## Search opponent

The search policy samples hidden-card determinizations from the acting player's information set, evaluates legal root actions through simulated continuations, and chooses the highest-valued action. `--search-rollouts` controls its sampling budget. This implementation is a root determinization search, not a claim of optimal Durak play.

## Training

The trainer uses legal-action-masked policy gradients, generalized advantage estimates, a value loss, entropy regularization, gradient clipping, and recurrent-state truncation. Rollout graphs are currently retained until the update; large batches can therefore consume substantial memory.

The optional hard-label teacher is disabled by default. It remains available for reproducing the negative experiment, but should not be treated as the recommended configuration.
