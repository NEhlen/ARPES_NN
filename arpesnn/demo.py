"""Generate a small, deterministic synthetic example; optionally denoise it."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from corpus import simulate
from band_models import FAMILIES
from denoise import load_denoiser, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/demo"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--family", choices=FAMILIES, default="avoided_crossing")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--counts", type=float, default=5.0)
    args = p.parse_args()
    if args.counts <= 0 or not np.isfinite(args.counts):
        p.error("Counts must be finite and positive.")
    if args.output.exists():
        p.error("Output already exists; choose a new directory.")
    images, grids, _, _, metadata = simulate(args.seed, args.family)
    clean = images[0]
    noisy = (
        np.random.default_rng(args.seed + 1).poisson(clean * args.counts) / args.counts
    ).astype("float32")
    args.output.mkdir(parents=True)
    np.save(args.output / "noisy.npy", noisy)
    np.save(args.output / "clean.npy", clean)
    np.save(args.output / "grids.npy", grids)
    arrays = [noisy, clean]
    titles = ["Noisy synthetic input", "Clean target"]
    if args.checkpoint:
        torch.set_num_threads(2)
        device = torch.device("cpu")
        _, pred, _ = predict(noisy, load_denoiser(args.checkpoint, device), device)
        arrays.append(pred)
        titles.append("Denoised")
        np.save(args.output / "denoised.npy", pred)
    fig, axes = plt.subplots(
        1, len(arrays), figsize=(4 * len(arrays), 4), layout="constrained"
    )
    for ax, array, title in zip(axes, arrays, titles):
        ax.imshow(
            array,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=np.percentile(clean, 99.5),
        )
        ax.set(title=title, xlabel="Momentum pixel", ylabel="Energy pixel")
    fig.savefig(args.output / "comparison.png", dpi=140)
    plt.close(fig)
    metadata["demo_counts_per_mean_pixel"] = args.counts
    (args.output / "parameters.json").write_text(json.dumps(metadata, indent=2))
    print(f"Example written to {args.output}")


if __name__ == "__main__":
    main()
