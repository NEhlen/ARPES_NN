"""Train separate v2 models with fresh noise, fixed splits and resumable state."""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from compact_model import CompactUNet


class CorpusDataset(Dataset):
    def __init__(self, root, split, seed=42):
        self.root = Path(root)
        self.split = split
        self.seed = seed
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema_version"] != 2:
            raise ValueError("Expected a complete v2 corpus")
        self.info = self.manifest["splits"][split]
        self.epoch = mp.get_context("spawn").Value("i", 0)
        self.images = None
        self.counts = None

    def __len__(self):
        return self.info["count"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["images"] = state["counts"] = None
        return state

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.root / self.split / "images.npy", mmap_mode="r")
            self.counts = np.load(self.root / self.split / "counts.npy", mmap_mode="r")
        epoch = self.epoch.value if self.split == "train" else 0
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.info["first_seed"] + index, epoch])
        )
        clean, bare = self.images[index]
        counts = float(self.counts[index])
        noisy = (rng.poisson(clean.astype(np.float64) * counts) / counts).astype(
            np.float32
        )
        scale = float(noisy.mean())
        if scale <= 0:
            raise ValueError("Empty Poisson sample")
        x, denoise, bare = noisy / scale, clean / scale, np.array(bare)
        if self.split == "train" and rng.random() < 0.5:
            x, denoise, bare = (
                np.ascontiguousarray(a[:, ::-1]) for a in (x, denoise, bare)
            )
        return (
            torch.from_numpy(x[None]),
            torch.from_numpy(denoise[None]),
            torch.from_numpy(bare[None]),
            index,
        )


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def move_batch(batch, device):
    return [
        x.to(device, non_blocking=True, memory_format=torch.channels_last)
        for x in batch[:3]
    ]


def ridge_error(prediction, energies, bands):
    """Dominant predicted peak to nearest visible true band, not full band recovery."""
    index = prediction.argmax(axis=1)
    peak_energy = np.take_along_axis(energies, index, axis=1)
    valid = (
        np.isfinite(bands)
        & (bands >= energies[:, :1, None])
        & (bands <= np.minimum(energies[:, -1:, None], 0))
    )
    distance = np.where(valid, np.abs(peak_energy[:, None, :] - bands), np.inf).min(
        axis=1
    )
    usable = np.isfinite(distance)
    return float(distance[usable].sum()), int(usable.sum())


def evaluate(models, loader, device, physical=False):
    for model in models.values():
        model.eval()
    families = loader.dataset.info["families"]
    accum = {
        task: torch.zeros(3 + len(families), device=device, dtype=torch.float64)
        for task in models
    }
    family_counts = torch.zeros(len(families), device=device, dtype=torch.float64)
    seen = 0
    ridge_sum = ridge_count = input_ridge_sum = 0
    smoothing_error = 0.0
    if physical:
        folder = loader.dataset.root / loader.dataset.split
        grids = np.load(folder / "grids.npy", mmap_mode="r")
        truth_bands = np.load(folder / "bands.npy", mmap_mode="r")
    with torch.inference_mode():
        for batch in loader:
            x, denoise, bare = move_batch(batch, device)
            family_indices = batch[3].to(device) % len(families)
            family_counts.scatter_add_(
                0, family_indices, torch.ones_like(family_indices, dtype=torch.float64)
            )
            seen += len(x)
            for task, model in models.items():
                target = denoise if task == "denoise" else bare
                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.float16,
                ):
                    prediction = model(x)
                diff = prediction.float() - target
                per_sample = diff.square().mean((1, 2, 3))
                accum[task][0] += per_sample.sum()
                accum[task][1] += diff.abs().mean((1, 2, 3)).sum()
                accum[task][2] += (x - target).square().mean((1, 2, 3)).sum()
                accum[task][3:].scatter_add_(0, family_indices, per_sample.double())
                if physical and task == "bareband":
                    indices = batch[3].numpy()
                    energy, bands = grids[indices, 0], truth_bands[indices]
                    error, count = ridge_error(
                        prediction.float().cpu().numpy()[:, 0], energy, bands
                    )
                    input_error, _ = ridge_error(batch[0].numpy()[:, 0], energy, bands)
                    ridge_sum += error
                    ridge_count += count
                    input_ridge_sum += input_error
                if physical and task == "denoise":
                    from scipy.ndimage import gaussian_filter

                    smooth = gaussian_filter(batch[0].numpy(), sigma=(0, 0, 1, 1))
                    smoothing_error += float(
                        np.square(smooth - batch[1].numpy()).mean((1, 2, 3)).sum()
                    )
    result = {}
    counts = family_counts.cpu().numpy()
    for task, values in accum.items():
        a = values.cpu().numpy()
        result[task] = {
            "mse": float(a[0] / seen),
            "mae": float(a[1] / seen),
            "input_mse": float(a[2] / seen),
            "family_mse": {
                f: float(a[3 + i] / counts[i])
                for i, f in enumerate(families)
                if counts[i]
            },
        }
    if physical and "denoise" in result:
        result["denoise"]["gaussian_sigma1_mse"] = smoothing_error / seen
    if physical and "bareband" in result and ridge_count:
        result["bareband"]["dominant_ridge_nearest_band_mae_eV"] = (
            ridge_sum / ridge_count
        )
        result["bareband"]["input_dominant_ridge_nearest_band_mae_eV"] = (
            input_ridge_sum / ridge_count
        )
        result["bareband"]["ridge_columns_evaluated"] = ridge_count
    return result


def save_prediction_preview(models, corpus, seed, device, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = [("test", 0), ("test", 3), ("ood", 0), ("ood", 1)]
    titles = [
        "Noisy input",
        "Clean target",
        "Denoised",
        "Bare target",
        "Bare prediction",
    ]
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), constrained_layout=True)
    for row, (split, index) in enumerate(examples):
        dataset = CorpusDataset(corpus, split, seed)
        index = min(index, len(dataset) - 1)
        x, clean, bare, _ = dataset[index]
        params = json.loads(
            (Path(corpus) / split / "parameters.jsonl").read_text().splitlines()[index]
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.float16,
            ),
        ):
            predictions = {
                task: model(x[None].to(device)).float().cpu().numpy()[0, 0]
                for task, model in models.items()
            }
        arrays = [
            x[0].numpy(),
            clean[0].numpy(),
            predictions.get("denoise"),
            bare[0].numpy(),
            predictions.get("bareband"),
        ]
        w = params["window"]
        for col, (title, array) in enumerate(zip(titles, arrays)):
            ax = axes[row, col]
            if array is None:
                ax.axis("off")
                continue
            vmax = np.percentile(arrays[1 if col < 3 else 3], 99.5)
            ax.imshow(
                array,
                origin="lower",
                aspect="auto",
                extent=[w["smin"], w["smax"], w["emin"], w["emax"]],
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            ax.set(
                title=title,
                xlabel="Cut coordinate (Å⁻¹)",
                ylabel=f"{split}: {params['family']}\nE − EF (eV)",
            )
    fig.suptitle(
        "Held-out synthetic predictions — shared scales within each task; not experimental validation"
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_loader(dataset, batch_size, workers, shuffle=False, generator=None):
    options = (
        {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        **options,
    )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA required; launch with host GPU access outside the restricted sandbox"
        )
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.patience < 1
        or args.lr <= 0
    ):
        raise ValueError("Invalid training sizes")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    root = Path(args.output)
    if root.exists() and not args.resume:
        raise FileExistsError("Output exists; choose a new output or use --resume")
    root.mkdir(parents=True, exist_ok=True)
    tasks = ("denoise", "bareband") if args.task == "both" else (args.task,)
    datasets = {
        split: CorpusDataset(args.corpus, split, args.seed)
        for split in ("train", "validation")
    }
    generator = torch.Generator()
    loaders = {
        split: make_loader(
            data,
            args.batch_size,
            args.workers,
            split == "train",
            generator if split == "train" else None,
        )
        for split, data in datasets.items()
    }
    manifest = datasets["train"].manifest
    corpus_hash = hashlib.sha256(
        (Path(args.corpus) / "manifest.json").read_bytes()
    ).hexdigest()
    config = {
        **vars(args),
        "corpus": str(Path(args.corpus).resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "corpus_manifest_sha256": corpus_hash,
        "architecture": "compact_unet_v1",
        "base_channels": 16,
        "input_channels": 1,
        "input_shape": [manifest["size"], manifest["size"]],
        "normalization": "divide_intensity_by_input_mean",
        "noise": "fresh_per_epoch_train_fixed_validation",
        "tasks": list(tasks),
    }
    if args.resume:
        old = json.loads((root / "run_config.json").read_text())
        for key in (
            "corpus_manifest_sha256",
            "seed",
            "architecture",
            "tasks",
            "batch_size",
        ):
            if old[key] != config[key]:
                raise ValueError(f"Resume config mismatch: {key}")
    else:
        snapshot = root / "source"
        snapshot.mkdir()
        source_files = (
            "train_tasks.py",
            "compact_model.py",
            "corpus.py",
            "band_models.py",
        )
        config["source_sha256"] = {}
        for name in source_files:
            source = Path(__file__).with_name(name)
            shutil.copy2(source, snapshot / name)
            config["source_sha256"][name] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        atomic_json(root / "run_config.json", config)
    models = {
        task: CompactUNet(task).to(device=device, memory_format=torch.channels_last)
        for task in tasks
    }
    optimizers = {
        task: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for task, model in models.items()
    }
    scalers = {
        task: torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for task in tasks
    }
    schedulers = {
        task: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        for task, opt in optimizers.items()
    }
    best = {task: float("inf") for task in tasks}
    stale = {task: 0 for task in tasks}
    start_epoch = 0
    if args.resume:
        saved = torch.load(root / "last.pt", map_location=device, weights_only=False)
        for task in tasks:
            models[task].load_state_dict(saved["models"][task])
            optimizers[task].load_state_dict(saved["optimizers"][task])
            scalers[task].load_state_dict(saved["scalers"][task])
            schedulers[task].load_state_dict(saved["schedulers"][task])
        best, stale, start_epoch = saved["best"], saved["stale"], saved["epoch"]
    for task in tasks:
        folder = root / task
        folder.mkdir(exist_ok=True)
        atomic_json(folder / "best_model.metadata.json", dict(config, task=task))
    metrics_path = root / "metrics.csv"
    if not args.resume:
        metrics_path.write_text(
            "epoch,task,train_mse,validation_mse,validation_input_mse,seconds,lr,best_validation_mse\n"
        )
    print(json.dumps(config), flush=True)
    began = time.perf_counter()
    if not args.resume:
        initial = evaluate(models, loaders["validation"], device)
        atomic_json(root / "initial_validation.json", initial)
        for task in tasks:
            best[task] = initial[task]["mse"]
            atomic_torch_save(
                {
                    "architecture": "compact_unet_v1",
                    "task": task,
                    "base_channels": 16,
                    "input_channels": 1,
                    "input_shape": config["input_shape"],
                    "state_dict": {
                        k: v.detach().cpu()
                        for k, v in models[task].state_dict().items()
                    },
                    "epoch": 0,
                    "validation": initial[task],
                    "corpus_manifest_sha256": corpus_hash,
                },
                root / task / "best_model.pt",
            )
        print(
            f"Initial validation (denoiser = identity): {json.dumps(initial)}",
            flush=True,
        )
    active = [task for task in tasks if stale[task] < args.patience]
    for epoch in range(start_epoch, args.epochs):
        if not active:
            break
        started = time.perf_counter()
        datasets["train"].epoch.value = epoch
        generator.manual_seed(args.seed + epoch)
        totals = {task: torch.zeros((), device=device) for task in active}
        seen = 0
        for task in active:
            models[task].train()
        for step, batch in enumerate(loaders["train"], 1):
            x, denoise, bare = move_batch(batch, device)
            seen += len(x)
            for task in active:
                target = denoise if task == "denoise" else bare
                optimizer, scaler = optimizers[task], scalers[task]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.float16,
                ):
                    output = models[task](x)
                    loss = torch.nn.functional.mse_loss(output, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite {task} loss at epoch {epoch+1}, batch {step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(models[task].parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                totals[task] += loss.detach() * len(x)
            if step == 1 or step % 50 == 0 or step == len(loaders["train"]):
                progress = {task: float(totals[task] / seen) for task in active}
                status = {
                    "state": "training",
                    "pid": os.getpid(),
                    "epoch": epoch + 1,
                    "epochs": args.epochs,
                    "batch": step,
                    "batches": len(loaders["train"]),
                    "active_tasks": active,
                    "running_train_mse": progress,
                    "elapsed_seconds": time.perf_counter() - began,
                }
                atomic_json(root / "status.json", status)
                print(json.dumps(status), flush=True)
        validation = evaluate(
            {t: models[t] for t in active}, loaders["validation"], device
        )
        seconds = time.perf_counter() - started
        for task in active:
            val = validation[task]["mse"]
            if not np.isfinite(val):
                raise FloatingPointError("Nonfinite validation loss")
            schedulers[task].step(val)
            if val < best[task]:
                best[task], stale[task] = val, 0
                checkpoint = {
                    "architecture": "compact_unet_v1",
                    "task": task,
                    "base_channels": 16,
                    "input_channels": 1,
                    "input_shape": config["input_shape"],
                    "state_dict": {
                        k: v.detach().cpu()
                        for k, v in models[task].state_dict().items()
                    },
                    "epoch": epoch + 1,
                    "validation": validation[task],
                    "corpus_manifest_sha256": corpus_hash,
                }
                atomic_torch_save(checkpoint, root / task / "best_model.pt")
            else:
                stale[task] += 1
            with metrics_path.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch + 1,
                        task,
                        float(totals[task] / seen),
                        val,
                        validation[task]["input_mse"],
                        seconds,
                        optimizers[task].param_groups[0]["lr"],
                        best[task],
                    ]
                )
        atomic_torch_save(
            {
                "epoch": epoch + 1,
                "models": {t: models[t].state_dict() for t in tasks},
                "optimizers": {t: optimizers[t].state_dict() for t in tasks},
                "scalers": {t: scalers[t].state_dict() for t in tasks},
                "schedulers": {t: schedulers[t].state_dict() for t in tasks},
                "best": best,
                "stale": stale,
            },
            root / "last.pt",
        )
        print(
            f"epoch {epoch+1} done in {seconds:.1f}s; validation={json.dumps(validation)}",
            flush=True,
        )
        active = [task for task in tasks if stale[task] < args.patience]
    # Evaluate the selected best models on untouched ID and unseen-family tests.
    for task in tasks:
        checkpoint = torch.load(
            root / task / "best_model.pt", map_location=device, weights_only=True
        )
        models[task].load_state_dict(checkpoint["state_dict"])
    report = {}
    for split in ("test", "ood"):
        dataset = CorpusDataset(args.corpus, split, args.seed)
        report[split] = evaluate(
            models,
            make_loader(dataset, args.batch_size, args.workers),
            device,
            physical=True,
        )
    atomic_json(root / "evaluation.json", report)
    save_prediction_preview(
        models, args.corpus, args.seed, device, root / "heldout_predictions.png"
    )
    atomic_json(
        root / "status.json",
        {
            "state": "complete",
            "pid": os.getpid(),
            "best_validation_mse": best,
            "elapsed_seconds": time.perf_counter() - began,
            "evaluation": report,
        },
    )
    print(f"Training complete. Evaluation: {json.dumps(report)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--task", choices=("both", "denoise", "bareband"), default="both"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        train(args)
    except Exception as error:
        root = Path(args.output)
        if root.exists() and not isinstance(error, FileExistsError):
            atomic_json(
                root / "status.json",
                {"state": "failed", "error": repr(error), "pid": os.getpid()},
            )
        raise


if __name__ == "__main__":
    main()
