"""Synthetic additive-background task; intrinsic spectral intensity remains signal."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from functools import lru_cache
import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
import torch
from torch.utils.data import Dataset
from corpus import simulate
from band_models import TRAIN_FAMILIES, HELD_OUT_FAMILIES


@lru_cache(maxsize=4)
def grid(size):
    x = np.linspace(-1, 1, size, dtype=np.float32)
    return x[:, None], x[None, :]


def make_background(signal, rng, kind=None):
    """Return independent additive intensity; no intrinsic component is relabelled."""
    e, k = grid(signal.shape[0])
    kinds = ["none", "constant", "tilted", "hump", "step", "correlated", "mixture"]
    kind = kind or str(rng.choice(kinds, p=[0.2, 0.1, 0.15, 0.15, 0.1, 0.15, 0.15]))
    if kind not in kinds:
        raise ValueError("Unknown background kind")
    ratio = 0.0 if kind == "none" else float(10 ** rng.uniform(-1.7, 0.5))
    offset = float(rng.uniform(-0.6, 0.6))
    width = float(rng.uniform(0.2, 1.2))
    flat = np.ones_like(signal)
    tilted = np.exp(rng.uniform(-0.8, 0.8) * e + rng.uniform(-0.4, 0.4) * k) * flat
    hump = (0.2 + np.exp(-0.5 * ((e - offset) / width) ** 2)) * (
        1 + rng.uniform(-0.3, 0.3) * k
    )
    step = (0.15 + expit((e - offset) / rng.uniform(0.08, 0.35))) * (
        1 + rng.uniform(-0.25, 0.25) * k
    )
    # Loss-like, signal-correlated energy tail; illustrative, not a calibrated Shirley model.
    profile = gaussian_filter1d(signal.mean(1), signal.shape[0] * 0.04)
    correlated = (
        0.1 + np.cumsum(profile[::-1])[::-1] / max(float(profile.sum()), 1e-8)
    )[:, None] * (1 + rng.uniform(-0.2, 0.2) * k)
    choices = {
        "constant": flat,
        "tilted": tilted,
        "hump": hump,
        "step": step,
        "correlated": correlated,
        "mixture": 0.4 * tilted + 0.3 * hump + 0.3 * correlated,
        "none": flat,
    }
    b = np.broadcast_to(choices[kind], signal.shape).copy()
    # The same cropped angular acceptance applies to signal and background.
    b *= np.any(signal > 0, axis=0)[None, :]
    b *= ratio * float(signal.mean()) / max(float(b.mean()), 1e-8)
    return b.astype("float32"), {"kind": kind, "background_to_signal_mean": ratio}


V2_REGIMES = (
    "zero",
    "mean_weak",
    "mean_comparable",
    "mean_dominant",
    "local_high",
    "local_comparable",
    "local_buried",
)


def local_band_mask(signal):
    """Signal-only ridge neighbourhood; excludes nearly empty momentum columns."""
    peaks = signal.max(axis=0)
    return (signal >= 0.5 * peaks[None, :]) & (peaks[None, :] >= 0.1 * peaks.max())


def make_background_v2(signal, rng, regime):
    if regime not in V2_REGIMES:
        raise ValueError(regime)
    kind = (
        "none"
        if regime == "zero"
        else str(
            rng.choice(["constant", "tilted", "hump", "step", "correlated", "mixture"])
        )
    )
    b, meta = make_background(signal, rng, kind)
    mask = local_band_mask(signal)
    if regime.startswith("mean_"):
        bounds = {
            "mean_weak": (0.05, 0.5),
            "mean_comparable": (0.5, 2),
            "mean_dominant": (2, 10),
        }[regime]
        ratio = 10 ** rng.uniform(*np.log10(bounds))
        b *= ratio * signal.mean() / b.mean()
    elif regime.startswith("local_"):
        bounds = {
            "local_high": (2, 10),
            "local_comparable": (0.5, 2),
            "local_buried": (0.05, 0.5),
        }[regime]
        contrast = 10 ** rng.uniform(*np.log10(bounds))
        b *= np.median(signal[mask] / np.maximum(b[mask], 1e-20)) / contrast
    meta.update(
        regime=regime,
        background_to_signal_mean=float(b.mean() / signal.mean()),
        local_signal_to_background=(
            float(np.median(signal[mask] / np.maximum(b[mask], 1e-20)))
            if regime != "zero"
            else -1.0
        ),
    )
    return b, meta


def signal_job(job):
    seed, family, size = job
    images, grids, _, _, meta = simulate(seed, family, size, include_background=False)
    signal = images[0]
    rng = np.random.default_rng(seed + 92871)
    e, k = grid(size)
    feature = str(
        rng.choice(["unchanged", "broad_flat", "weak_band"], p=[0.5, 0.25, 0.25])
    )
    if feature != "unchanged":
        center = rng.uniform(-0.6, 0.6) + rng.uniform(-0.08, 0.08) * k
        width = (
            rng.uniform(0.10, 0.3)
            if feature == "broad_flat"
            else rng.uniform(0.025, 0.10)
        )
        extra = width**2 / ((e - center) ** 2 + width**2)
        amplitude = (
            rng.uniform(0.15, 0.65)
            if feature == "broad_flat"
            else rng.uniform(0.02, 0.12)
        )
        extra *= np.any(signal > 0, axis=0)[None, :]
        signal = signal + amplitude * signal.mean() * extra / max(
            float(extra.mean()), 1e-8
        )
        signal /= signal.mean()
    meta["retained_signal_feature"] = feature
    meta["task"] = "additive_background_v1"
    return signal.astype("float32"), grids, meta


def create(root, counts, size=256, workers=4):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    if size < 32 or size % 4 or min(counts.values()) < 1:
        raise ValueError(
            "Positive split sizes and image size >=32 divisible by four required"
        )
    root.mkdir(parents=True)
    manifest = {"schema": "additive_background_v1", "size": size, "splits": {}}
    with mp.get_context("spawn").Pool(workers) as pool:
        for j, (split, n) in enumerate(counts.items(), 1):
            families = HELD_OUT_FAMILIES if split == "ood" else TRAIN_FAMILIES
            first = 10_000_000 + j * 1_000_000
            folder = root / split
            folder.mkdir()
            signals = open_memmap(
                folder / "signal.npy", mode="w+", dtype="float32", shape=(n, size, size)
            )
            grids = open_memmap(
                folder / "grids.npy", mode="w+", dtype="float32", shape=(n, 3, size)
            )
            with (folder / "parameters.jsonl").open("w") as f:
                jobs = [
                    (first + i, families[i % len(families)], size) for i in range(n)
                ]
                for i, (signal, g, meta) in enumerate(
                    pool.imap(signal_job, jobs, chunksize=4)
                ):
                    signals[i] = signal
                    grids[i] = g
                    f.write(json.dumps(meta) + "\n")
                    if (i + 1) % 500 == 0:
                        print(f"{split}: {i+1}/{n}", flush=True)
            signals.flush()
            grids.flush()
            manifest["splits"][split] = {
                "count": n,
                "first_seed": first,
                "families": list(families),
                "parameters_sha256": hashlib.sha256(
                    (folder / "parameters.jsonl").read_bytes()
                ).hexdigest(),
            }
    manifest["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ["background_data.py", "corpus.py", "band_models.py"]
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Corpus ready: {root}", flush=True)


class BackgroundDataset(Dataset):
    def __init__(self, root, split, seed=42, profile="v1"):
        if profile not in ("v1", "v2"):
            raise ValueError(profile)
        self.profile = profile
        self.root = Path(root)
        self.split = split
        self.seed = seed
        self.epoch = 0
        self.signals = None
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != "additive_background_v1":
            raise ValueError("Wrong corpus schema")
        self.info = self.manifest["splits"][split]

    def __len__(self):
        return self.info["count"]

    def __getitem__(self, index):
        if self.signals is None:
            self.signals = np.load(self.root / self.split / "signal.npy", mmap_mode="r")
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.seed,
                    self.info["first_seed"] + index,
                    self.epoch if self.split == "train" else 0,
                ]
            )
        )
        signal = np.array(self.signals[index])
        if self.profile == "v2":
            regime = V2_REGIMES[
                (index + (self.epoch if self.split == "train" else 0)) % len(V2_REGIMES)
            ]
            background, meta = make_background_v2(signal, rng, regime)
        else:
            background, meta = make_background(signal, rng)
        counts = float(10 ** rng.uniform(0.5, 3.3))
        if self.profile == "v2":
            # Hold total mean counts fixed: stronger background must not buy more photons.
            counts /= float((signal + background).mean())
        noisy = (
            rng.poisson((signal + background).astype("float64") * counts) / counts
        ).astype("float32")
        scale = float(noisy.mean())
        if self.split == "train" and rng.random() < 0.5:
            signal, background, noisy = [
                np.ascontiguousarray(a[:, ::-1]) for a in (signal, background, noisy)
            ]
        return {
            "input": torch.from_numpy(noisy[None] / scale),
            "background": torch.from_numpy(background[None] / scale),
            "signal": torch.from_numpy(signal[None] / scale),
            "zero": meta["kind"] == "none",
            "index": index,
            "kind": meta["kind"],
            "counts": counts,
            "regime": meta.get("regime", "v1"),
            "background_to_signal_mean": float(background.mean() / signal.mean()),
            "local_signal_to_background": meta.get("local_signal_to_background", -1.0),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--validation", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--ood", type=int, default=1000)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    create(
        a.output,
        {s: getattr(a, s) for s in ["train", "validation", "test", "ood"]},
        a.size,
        a.workers,
    )


if __name__ == "__main__":
    main()
