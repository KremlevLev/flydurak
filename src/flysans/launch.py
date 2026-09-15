import argparse
import hashlib
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import torch

CHECKPOINT_URL = "https://github.com/KremlevLev/flydurak/releases/download/v0.1.0/durak-fast-plastic-r32.best.pt"
CHECKPOINT_SHA256 = "e9b0388e8aea208ee509f67a04974de563e27ac27090808ebe62327653480285"


def _first_existing(candidates):
    return next((path for path in candidates if path.is_file()), None)


def _free_port(host, preferred):
    for port in range(preferred, preferred + 100):
        with socket.socket() as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in {preferred}..{preferred + 99}")


def _open_browser(url):
    time.sleep(1.2)
    webbrowser.open(url)


def _download_checkpoint(root):
    target = root / "models/durak-fast-plastic-r32.best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    print("Checkpoint is missing; downloading 374 MiB from the FlyDurak release...", flush=True)

    def progress(blocks, block_size, total):
        received = min(blocks * block_size, total)
        if total > 0:
            print(f"\rDownloading weights: {received / total:6.1%}", end="", flush=True)

    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, temporary, progress)
        print(flush=True)
        checksum = hashlib.sha256()
        with temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError(f"checkpoint checksum mismatch: {digest}")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target

def _prepare_graph(root):
    from .prepare import build, download
    raw = root / "data/raw/malecns-v1.0"
    target = root / "data/processed/malecns-v1.0-soma.pt"
    print("MaleCNS graph is missing; downloading and building it once...", flush=True)
    download(raw)
    build(raw, target, minimum_weight=3)
    return target


def main():
    parser = argparse.ArgumentParser(description="Start the FlyDurak browser game")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="preferred port; the next free port is used automatically")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading a missing checkpoint")
    args = parser.parse_args()

    root = Path.cwd()
    graph = args.graph or _first_existing((
        root / "data/processed/malecns-v1.0-soma.pt",
        root / "data/processed/malecns-v1.0.pt",
    ))
    checkpoint = args.checkpoint or _first_existing((
        root / "models/durak-fast-plastic-r32.best.pt",
        root / "runs/durak-fast-plastic-r32.best.pt",
        root / "runs/durak-deep-league.best.pt",
        root / "runs/durak-league-2.best.pt",
    ))
    if graph is None:
        if args.no_download:
            parser.error("MaleCNS graph not found under data/processed")
        graph = _prepare_graph(root)
    if checkpoint is None:
        if args.no_download:
            parser.error("checkpoint not found under models/ or runs/")
        checkpoint = _download_checkpoint(root)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but this PyTorch build cannot use CUDA; use --device cpu or auto")
    port = _free_port(args.host, args.port)
    public_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{public_host}:{port}/"
    print(f"FlyDurak: {device.upper()} | {graph.name} | port {port}", flush=True)
    print(f"Open: {url}", flush=True)
    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    from .durak_web import main as web_main
    sys.argv = [sys.argv[0], "--device", device, "--graph", str(graph), "--checkpoint", str(checkpoint), "--host", args.host, "--port", str(port)]
    web_main()


if __name__ == "__main__":
    main()
