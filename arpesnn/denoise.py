"""Apply a compact denoising checkpoint to a 2D NumPy array or corrected SP2 image."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compact_model import CompactUNet
from apply_model import load_experimental_file, resize_array


def load_denoiser(checkpoint, device):
    bundle = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        bundle.get("architecture") != "compact_unet_v1"
        or bundle.get("task") != "denoise"
    ):
        raise ValueError(
            "Expected a compact_unet_v1 denoise checkpoint, not a bare-band or legacy model."
        )
    model = CompactUNet("denoise", bundle["base_channels"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    return model.eval()


def predict(data, model, device, size=256):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or min(data.shape) < 2 or not np.isfinite(data).all():
        raise ValueError(
            "Input must be a finite 2D array with energy rows and momentum/angle columns."
        )
    if data.min() < 0 or data.mean() <= 0:
        raise ValueError(
            "Input must be nonnegative with positive mean; background-subtracted negative spectra are outside this model's training distribution."
        )
    if size < 16 or size % 4:
        raise ValueError("Model grid size must be >=16 and divisible by four.")
    sampled = resize_array(data, (size, size))
    scale = float(sampled.mean())
    with torch.inference_mode():
        output = (
            model(torch.from_numpy(sampled / scale)[None, None].to(device))[0, 0]
            .cpu()
            .numpy()
            * scale
        )
    return sampled, output, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory (never overwrites an existing directory).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Inference grid; released model trained at 256. Changing it changes feature scales.",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    torch.set_num_threads(2)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")
    if args.input.suffix.lower() == ".npy":
        raw = np.load(args.input, allow_pickle=False)
        meta = {"x_label": "Momentum / angle pixel", "y_label": "Energy pixel"}
    else:
        raw, meta = load_experimental_file(args.input)
    model = load_denoiser(args.checkpoint, device)
    sampled, prediction, scale = predict(raw, model, device, args.size)
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "prediction.npz",
        raw=raw,
        model_input=sampled,
        prediction=prediction,
        residual=sampled - prediction,
    )
    np.save(args.output / "denoised.npy", prediction)
    extent = [
        meta.get("xmin", 0),
        meta.get("xmax", raw.shape[1] - 1),
        meta.get("ymin", 0),
        meta.get("ymax", raw.shape[0] - 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    lo, hi = np.percentile(sampled, [1, 99.5])
    for ax, array, title in zip(
        axes,
        (raw, sampled, prediction),
        ("Original", "Resampled model input", "Denoised"),
    ):
        im = ax.imshow(
            array,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=lo,
            vmax=hi,
        )
        ax.set(
            title=title,
            xlabel=meta.get("x_label", "Momentum pixel"),
            ylabel=meta.get("y_label", "Energy pixel"),
        )
    fig.colorbar(im, ax=axes, label="Input intensity units", shrink=0.8)
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    metadata = {
        "input": args.input.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": str(device),
        "normalization_mean": scale,
        "original_shape": list(raw.shape),
        "model_shape": list(prediction.shape),
        "output_units": "same intensity units as input",
        "resampling": "bilinear to model grid; no original-resolution claim",
        "source_metadata": meta,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Wrote {args.output}; output is {args.size} x {args.size}, in input intensity units."
    )


if __name__ == "__main__":
    main()
