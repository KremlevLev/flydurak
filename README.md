# FlyDurak

**English** | [Русский](README.ru.md)

A published **MaleCNS fruit-fly connectome** turned into a recurrent policy for two-player Durak. The playable model uses **165,122 neurons and 10,511,038 directed edges** and shows its strongest neural signals while choosing a card.

> Fun research demo, not evidence that a fly brain understands cards or that biological topology beats conventional neural networks.

## Run it

Python 3.11+ is the only prerequisite. From the repository directory:

```bash
python run.py
```

On Windows, `py run.py` works too. The first run creates `.venv`, installs the project, downloads and builds MaleCNS, downloads and verifies the 374 MiB checkpoint, selects CUDA or CPU, finds a free port, and opens the game. Later runs start directly.

Clone and run in one line on Windows PowerShell (including PowerShell 5):

```powershell
git clone https://github.com/KremlevLev/flydurak.git; Set-Location flydurak; py run.py
```
or
```powershell
git clone https://github.com/KremlevLev/flydurak.git; Set-Location flydurak; py run.py
```
On Linux or macOS:

```bash
git clone https://github.com/KremlevLev/flydurak.git && cd flydurak && python3 run.py
```

### Docker

```bash
docker compose up --build
```

Then open [http://localhost:8765](http://localhost:8765). Data and weights are cached in `data/` and `models/`.

## What was trained

MaleCNS is converted to a signed, normalized sparse recurrent graph. The agent combines connectome state with a compact decision GRU, legal-action masking, actor/critic heads, and factorized source/target synaptic gains. It was trained with batched reinforcement learning against random, scripted, frozen-policy, and determinized-search opponents, followed by search-focused fine-tuning.

The browser uses official MaleCNS soma coordinates and visualizes strong hidden-state activations and directed synaptic signals. The full graph runs at inference; only a readable subset of signals is drawn.

The checkpoint is an exploratory demo: it can beat humans, but it is not a solved-game agent. A controlled multi-seed comparison with GRU/LSTM baselines remains unfinished.

## Options

```bash
python run.py --device cpu
python run.py --device cuda
python run.py --port 9000
python run.py --no-browser
```

More: [architecture](docs/ARCHITECTURE.md), [experiments](docs/EXPERIMENTS.md), [model card](docs/MODEL_CARD.md), [data](data/README.md).

Code: [MIT](LICENSE). MaleCNS is downloaded from its original public source and retains its own attribution and license requirements.
