"""Launch detached GPU background training with automatic post-training analysis."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experimental", type=Path)
    p.add_argument("--profile", choices=["v1", "v2"], default="v1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA unavailable here; use a GPU-accessible host terminal.")
    if args.output.exists():
        p.error("Choose a new output directory.")
    manifest = args.corpus / "manifest.json"
    if (
        not manifest.exists()
        or json.loads(manifest.read_text()).get("schema") != "additive_background_v1"
    ):
        p.error("Background corpus is missing or incomplete.")
    if args.experimental and not args.experimental.is_file():
        p.error("Experimental input does not exist.")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train_background.py").resolve()),
        "--corpus",
        str(args.corpus.resolve()),
        "--output",
        str(output),
        "--profile",
        args.profile,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--require-cuda",
    ]
    if args.experimental:
        command.extend(["--experimental", str(args.experimental.resolve())])
    log = output.with_suffix(".log")
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd=Path(__file__).resolve().parents[1],
            env=dict(
                os.environ,
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="1",
                MPLCONFIGDIR="/tmp/arpes-mpl",
            ),
        )
    info = {
        "pid": child.pid,
        "command": command,
        "log": str(log),
        "output": str(output),
        "after_training": "Held-out analysis, repeated component-fitting stress test, and optional experimental preview run automatically in this process.",
    }
    output.with_suffix(".launch.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
