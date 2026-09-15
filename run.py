"""One-command FlyDurak bootstrap and launcher."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def ready():
    try:
        import flysans  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

def main():
    os.chdir(ROOT)
    if ready():
        from flysans.launch import main as launch
        launch()
        return
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating .venv...", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", "--system-site-packages", str(environment)])
    print("Installing FlyDurak...", flush=True)
    subprocess.check_call([str(python), "-m", "pip", "install", "-e", "."])
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])

if __name__ == "__main__":
    main()
