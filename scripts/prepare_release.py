"""Prepare (but do not publish) a portable inference-weight archive."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = args.output.with_suffix(".tar.gz")
    if args.output.exists() or archive.exists():
        p.error("Output already exists; choose a new name.")
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        bundle.get("architecture") != "compact_unet_v1"
        or bundle.get("task") != "denoise"
    ):
        p.error("Expected a compact denoiser checkpoint.")
    args.output.mkdir(parents=True)
    shutil.copy2(args.checkpoint, args.output / "denoiser.pt")
    metadata = {
        key: bundle[key]
        for key in (
            "architecture",
            "task",
            "base_channels",
            "input_channels",
            "input_shape",
            "epoch",
            "validation",
            "corpus_manifest_sha256",
        )
        if key in bundle
    }
    metadata["normalization"] = (
        "divide by resampled input mean; multiply prediction by same mean"
    )
    metadata["checkpoint_sha256"] = hashlib.sha256(
        args.checkpoint.read_bytes()
    ).hexdigest()
    metadata["license"] = (
        "See LICENSE"
        if (root / "LICENSE").exists()
        else "Not selected; no reuse license granted by this archive"
    )
    (args.output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(root / "docs/MODEL_CARD.md", args.output / "MODEL_CARD.md")
    if (root / "LICENSE").exists():
        shutil.copy2(root / "LICENSE", args.output / "LICENSE")
    (args.output / "README.txt").write_text(
        "Inference weights for https://github.com/NEhlen/ARPES_NN\nUse arpesnn/denoise.py --checkpoint <this folder>/denoiser.pt.\nSee the repository docs for training and evaluation details.\nModel-card relative links refer to repository docs.\n"
        + metadata["license"]
        + "\n"
    )
    checksums = [
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name
        for path in sorted(args.output.iterdir())
        if path.is_file()
    ]
    (args.output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(args.output, arcname=args.output.name)
    print(f"Prepared {archive}; nothing uploaded.")


if __name__ == "__main__":
    main()
