# FlyDurak

**English** | [Русский](README.ru.md)

## Quick play

After installation, start the browser game with one command:

```bash
flydurak
```

The launcher automatically selects CUDA when available (otherwise CPU), downloads the release checkpoint when it is missing, finds the graph, chooses a free port starting at `8765`, and opens the browser.

An experimental reinforcement-learning project that turns the published **MaleCNS fruit-fly connectome** into a recurrent policy for two-player podkidnoy Durak.

The controller uses the complete processed graph used in our experiments: **165,122 neurons and 10,511,038 directed edges** after retaining traced neurons and connections with at least three synapses. The graph runs as a sparse recurrent network. A compact decision GRU, legal-action masking, actor/critic heads, and bounded factorized synaptic plasticity make the biological topology usable for the card task.

> This is a pet/research project, not evidence that a biological connectome is better than standard neural networks. The current agent is entertaining and can beat humans, but the controlled multi-seed baseline study is unfinished.

## What is included

- Complete headless two-player 36-card Durak environment.
- MaleCNS downloader and graph preprocessing.
- Sparse connectome, deep hybrid, GRU, LSTM, RNN, and MLP policies.
- Search, scripted, mixed, and frozen-policy opponents.
- Factorized plasticity affecting every retained connectome edge.
- CLI and browser interface for playing against a checkpoint.
- Topology controls and an experimental study runner.
- Deterministic smoke tests.

No dataset, checkpoint, Steam installation, or proprietary game asset is committed.

## Install

Python 3.11+ and a CUDA-enabled PyTorch installation are recommended for the full connectome.

```bash
git clone <your-repository-url>
cd drosophila-git
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
python3 -m pytest -q tests
```

On Windows, activate with `.venv\Scripts\activate`.

## Prepare MaleCNS on the training machine

The command downloads the public flat-connectome Feather tables and builds the sparse graph locally:

```bash
flysans-prepare all \
  --raw data/raw/malecns-v1.0 \
  --output data/processed/malecns-v1.0.pt \
  --minimum-weight 3
```

Expected output for this configuration:

```text
neurons: 165122
edges:   10511038
```

The processed graph is approximately 202 MB and is intentionally ignored by Git.

## Train the release configuration

The strongest current public configuration uses the deep hybrid controller, factorized plasticity, and a determinized search opponent. Start conservatively and adjust the batch size for the machine.

```bash
flydurak-train \
  --device cuda \
  --model connectome \
  --architecture deep \
  --plastic-edges \
  --edge-learning-rate 0.0001 \
  --plasticity-weight 0.0001 \
  --graph data/processed/malecns-v1.0.pt \
  --opponent search \
  --search-rollouts 32 \
  --search-fraction 0.5 \
  --updates 120 \
  --batch-size 256 \
  --decisions 128 \
  --learning-rate 0.00001 \
  --entropy-weight 0.01 \
  --eval-games 1024 \
  --output runs/flydurak.pt
```

The trainer writes `runs/flydurak.pt` and selects `runs/flydurak.best.pt` using the rolling win rate against the training opponent.

The optional hard-label search teacher is experimental and disabled by default. Our first `--teacher-weight 0.1` run reduced playing strength; see [Experiments](docs/EXPERIMENTS.md).

## Play in a browser

Place the processed graph and a compatible checkpoint on the machine, then run:

To add the official MaleCNS soma coordinates to an existing processed graph
without rebuilding its edges:

```bash
flysans-prepare positions \
  --raw data/raw/malecns-v1.0 \
  --graph data/processed/malecns-v1.0.pt \
  --output data/processed/malecns-v1.0-soma.pt
```

The raw directory only needs the small official
`body-annotations-male-cns-v1.0-minconf-0.5.feather` file for this command.

```bash
flydurak-web \
  --device cuda \
  --graph data/processed/malecns-v1.0-soma.pt \
  --checkpoint models/durak-fast-plastic-r32.best.pt \
  --host 0.0.0.0 \
  --port 8765
```

Open `http://SERVER_IP:8765/`. Only expose the port where your network policy permits it; the demo has no authentication and is intended for a trusted network.

The browser UI groups the hand by suit, sorts each suit from 6 through ace, and
keeps the trump group at the far right. After every fly move it also displays
the strongest hidden-state activations and the policy's leading legal choices.
The CNS shape is a deterministic functional projection: neuron IDs and values
come from the model, but the checkpoint does not contain anatomical XYZ
coordinates, so the displayed positions are not anatomical locations.

For a terminal game:

```bash
flydurak-play \
  --device cuda \
  --graph data/processed/malecns-v1.0.pt \
  --checkpoint models/durak-fast-plastic-r32.best.pt
```

## Repository map

```text
src/flysans/durak.py              game rules and opponents
src/flysans/durak_train.py        actor/critic training loop
src/flysans/model.py              sparse connectome and baselines
src/flysans/prepare.py            MaleCNS download and preprocessing
src/flysans/durak_web.py          FastAPI browser game
src/flysans/study.py              comparative experiment runner
src/flysans/topology_controls.py  graph null controls
tests/                             smoke and invariant tests
docs/                              architecture, results, model card
```

## Reproducibility and claims

Current numbers are exploratory single-run measurements. They are useful for choosing a demo checkpoint but not for claiming superiority of biological topology. A defensible comparison requires multiple seeds, matched baselines, held-out opponents, topology rewires, and confidence intervals.

See [Architecture](docs/ARCHITECTURE.md), [Experiments](docs/EXPERIMENTS.md), [Model card](docs/MODEL_CARD.md), and [Data provenance](data/README.md).

## License

Project code is available under the [MIT License](LICENSE). MaleCNS data is not redistributed here and retains its own attribution and license requirements.
