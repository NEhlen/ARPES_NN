"""Train a separate additive-background estimator and automatically evaluate it."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from background_data import BackgroundDataset
from background_model import BackgroundNet


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def atomic_save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def loader(dataset, batch_size, workers, shuffle=False, generator=None):
    options = {"multiprocessing_context": "spawn"} if workers else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        **options,
    )


def objective(prediction, target, zero):
    mse = (prediction.float() - target).square().mean((1, 2, 3))
    # No-background controls carry extra cost to discourage needless subtraction.
    return (mse * (1 + 3 * zero.float())).mean()


def validate(model, data, device):
    sums = {
        "weighted_mse": 0.0,
        "background_mse": 0.0,
        "zero_background_mean_removed": 0.0,
        "zero_baseline_mse": 0.0,
    }
    n = 0
    nzero = 0
    model.eval()
    with torch.inference_mode():
        for batch in data:
            x = batch["input"].to(device)
            target = batch["background"].to(device)
            zero = batch["zero"].to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                pred = model(x)
            b = len(x)
            n += b
            nzero += int(zero.sum())
            sums["weighted_mse"] += float(objective(pred, target, zero)) * b
            sums["background_mse"] += float((pred.float() - target).square().mean()) * b
            sums["zero_baseline_mse"] += float(target.square().mean()) * b
            sums["zero_background_mean_removed"] += (
                float(pred.float()[zero].sum() / pred[0].numel()) if zero.any() else 0.0
            )
    return {
        k: (
            v / (nzero if k == "zero_background_mean_removed" else n)
            if (nzero if k == "zero_background_mean_removed" else n)
            else None
        )
        for k, v in sums.items()
    }


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required but unavailable in this environment")
    root = args.output
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    trainset = BackgroundDataset(args.corpus, "train", args.seed, args.profile)
    valset = BackgroundDataset(args.corpus, "validation", args.seed, args.profile)
    valdata = loader(valset, args.batch_size, args.workers)
    model = BackgroundNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    config = {
        **vars(args),
        "corpus": str(args.corpus.resolve()),
        "output": str(root.resolve()),
        "experimental": str(args.experimental.resolve()) if args.experimental else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "corpus_sha256": hashlib.sha256(
            (args.corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "architecture": "background_coarse_v1",
        "task": "background",
        "model_grid": 128,
        "background_grid": 32,
        "base_channels": 16,
        "loss": "background MSE; zero-background examples weighted 4x",
    }
    best = float("inf")
    stale = 0
    start = 0
    if args.resume:
        previous = json.loads((root / "config.json").read_text())
        for k in ["corpus_sha256", "seed", "batch_size", "profile"]:
            previous_value = previous.get(k, "v1" if k == "profile" else None)
            if previous_value != config[k]:
                raise ValueError(f"Resume mismatch: {k}")
        saved = torch.load(root / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start = saved["epoch"]
        best = saved["best"]
        stale = saved["stale"]
    else:
        (root / "source").mkdir()
        for name in [
            "background_model.py",
            "background_plots.py",
            "background_data.py",
            "train_background.py",
            "analyze_background.py",
            "apply_background.py",
            "corpus.py",
            "compact_model.py",
        ]:
            shutil.copy2(Path(__file__).with_name(name), root / "source" / name)
        atomic_json(root / "config.json", config)
        (root / "metrics.csv").write_text(
            "epoch,train_loss,validation_loss,background_mse,zero_mean_removed,seconds,lr\n"
        )
    begun = time.monotonic()
    for epoch in range(start, args.epochs):
        trainset.epoch = epoch
        data = loader(
            trainset,
            args.batch_size,
            args.workers,
            True,
            torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        total = 0.0
        n = 0
        for step, batch in enumerate(data, 1):
            x = batch["input"].to(device)
            target = batch["background"].to(device)
            zero = batch["zero"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ):
                pred = model(x)
                loss = objective(pred, target, zero)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(x)
            n += len(x)
            if step % 50 == 0 or step == len(data):
                atomic_json(
                    root / "status.json",
                    {
                        "state": "training",
                        "pid": os.getpid(),
                        "epoch": epoch + 1,
                        "epochs": args.epochs,
                        "batch": step,
                        "batches": len(data),
                        "train_loss": total / n,
                        "elapsed_seconds": time.monotonic() - begun,
                    },
                )
        val = validate(model, valdata, device)
        score = val["weighted_mse"]
        scheduler.step(score)
        if score < best:
            best = score
            stale = 0
            atomic_save(
                root / "best_model.pt",
                {
                    "architecture": "background_coarse_v1",
                    "task": "background",
                    "profile": args.profile,
                    "base_channels": 16,
                    "input_shape": [trainset.manifest["size"]] * 2,
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation": val,
                    "corpus_sha256": config["corpus_sha256"],
                },
            )
        else:
            stale += 1
        atomic_save(
            root / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best": best,
                "stale": stale,
            },
        )
        with (root / "metrics.csv").open("a") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    total / n,
                    score,
                    val["background_mse"],
                    val["zero_background_mean_removed"],
                    time.monotonic() - begun,
                    optimizer.param_groups[0]["lr"],
                ]
            )
        print(f"epoch {epoch+1}: {json.dumps(val)}", flush=True)
        if stale >= args.patience:
            break
    atomic_json(
        root / "status.json",
        {
            "state": "analyzing",
            "pid": os.getpid(),
            "best_validation_loss": best,
            "elapsed_seconds": time.monotonic() - begun,
        },
    )
    from analyze_background import analyze

    analysis_output = root / "analysis"
    if analysis_output.exists():
        analysis_output = root / f"analysis-{int(time.time())}"

    analyze(
        args.corpus,
        root / "best_model.pt",
        analysis_output,
        device,
        args.experimental,
        args.seed,
        stress=not args.skip_analysis_stress,
    )
    atomic_json(
        root / "status.json",
        {
            "state": "complete",
            "pid": os.getpid(),
            "best_validation_loss": best,
            "elapsed_seconds": time.monotonic() - begun,
            "analysis": str(analysis_output),
        },
    )
    print("Training and analysis complete.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=["v1", "v2"], default="v1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--skip-analysis-stress",
        action="store_true",
        help="Skip repeated fit stress test for short smoke runs",
    )
    p.add_argument("--experimental", type=Path)
    args = p.parse_args()
    try:
        train(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            atomic_json(
                args.output / "status.json",
                {"state": "failed", "error": repr(error), "pid": os.getpid()},
            )
        raise


if __name__ == "__main__":
    main()
