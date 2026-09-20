"""Launch a bounded, detached GPU run; logs and status survive the terminal."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA unavailable here. Run on the host with GPU access.")
    corpus, output = args.corpus.resolve(), args.output.resolve()
    if not (corpus / "manifest.json").is_file():
        parser.error("Corpus is incomplete or missing")
    if output.exists() and not args.resume:
        parser.error("Output exists; use a new output or --resume")
    if args.resume and not (output / "last.pt").is_file():
        parser.error("No resumable last.pt checkpoint found")
    if args.resume and (output / "status.json").exists():
        status = json.loads((output / "status.json").read_text())
        pid = status.get("pid")
        if (
            pid
            and Path(f"/proc/{pid}/cmdline").exists()
            and b"train_tasks.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
        ):
            parser.error(f"Training process {pid} is still running")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train_tasks.py").resolve()),
        "--corpus",
        str(corpus),
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--require-cuda",
    ]
    if args.resume:
        command.append("--resume")
    log = output.with_suffix(".log")
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR="/tmp/arpes-mpl",
    )
    with log.open("a" if args.resume else "x") as handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
            cwd=Path(__file__).resolve().parents[1],
        )
    launch = {
        "pid": process.pid,
        "command": command,
        "log": str(log),
        "output": str(output),
    }
    output.with_suffix(".launch.json").write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
