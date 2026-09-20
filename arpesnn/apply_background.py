"""Estimate and retain background separately; never clip the subtracted spectrum."""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from background_model import load_background
from background_plots import background_limit
from apply_model import load_experimental_file, resize_array


def apply(model, source, output, device, size=256):
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if source.suffix.lower() == ".npy":
        raw = np.load(source, allow_pickle=False)
        meta = {}
    else:
        raw, meta = load_experimental_file(source)
    if raw.ndim != 2 or not np.isfinite(raw).all() or raw.min() < 0 or raw.mean() <= 0:
        raise ValueError("Expected nonnegative finite 2D intensity with positive mean")
    sampled = resize_array(raw, (size, size))
    scale = float(sampled.mean())
    with torch.inference_mode():
        background = (
            model(torch.from_numpy(sampled / scale)[None, None].to(device))[0, 0]
            .cpu()
            .numpy()
            * scale
        )
    corrected = sampled - background
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "decomposition.npz",
        raw=raw,
        model_input=sampled,
        background=background,
        corrected=corrected,
    )
    plot_comparison(
        raw, sampled, background, corrected, meta, output / "comparison.png"
    )
    summary = {
        "input": str(source),
        "normalization_mean": scale,
        "mean_background_fraction": float(background.mean() / sampled.mean()),
        "negative_corrected_fraction": float((corrected < 0).mean()),
        "note": "Signed subtraction only. Background shot noise remains; no denoising or bare-band recovery. Experimental decomposition has no ground truth.",
        "source_metadata": meta,
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_comparison(raw, sampled, background, corrected, meta, path):
    extent = [
        meta.get("xmin", 0),
        meta.get("xmax", raw.shape[1] - 1),
        meta.get("ymin", 0),
        meta.get("ymax", raw.shape[0] - 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), layout="constrained")
    hi = np.percentile(sampled, 99.5)
    images = []
    for ax, array, title in zip(
        axes,
        [raw, sampled, background, corrected],
        ["Original", "Resampled input", "Estimated background", "Input − background"],
    ):
        im = ax.imshow(
            array,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis" if title == "Estimated background" else "magma",
            vmin=0,
            vmax=(
                background_limit(background) if title == "Estimated background" else hi
            ),
        )
        images.append(im)
        ax.set(
            title=title,
            xlabel=meta.get("x_label", "Momentum pixel"),
            ylabel=meta.get("y_label", "Energy pixel"),
        )
    fig.colorbar(
        images[2], ax=axes[2], label="Background scale (input units)", shrink=0.8
    )
    fig.colorbar(
        images[3],
        ax=[axes[i] for i in (0, 1, 3)],
        label="Signal scale (input units)",
        shrink=0.8,
    )
    fig.suptitle(
        "Background uses its own linear scale (viridis); spectra share a scale (magma)\n"
        "Negative corrected pixels display black but remain in numeric output"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, saved = load_background(args.checkpoint, device)
    summary = apply(model, args.input, args.output, device, saved["input_shape"][0])
    summary["checkpoint_sha256"] = hashlib.sha256(
        args.checkpoint.read_bytes()
    ).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
