"""Validate a completed corpus, including independent seeds and replayed examples."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from corpus import simulate


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    report = {}
    for split, info in manifest["splits"].items():
        folder = root / split
        parameters = (folder / "parameters.jsonl").read_bytes()
        if hashlib.sha256(parameters).hexdigest() != info["parameters_sha256"]:
            raise ValueError(f"Parameter checksum mismatch in {split}")
        rows = [json.loads(line) for line in parameters.splitlines()]
        if len(rows) != info["count"]:
            raise ValueError("Metadata count mismatch")
        seeds = {row["seed"] for row in rows}
        if len(seeds) != len(rows) or seeds & seen:
            raise ValueError("Duplicate or leaking configuration seeds")
        seen |= seeds
        images = np.load(folder / "images.npy", mmap_mode="r")
        size = manifest["size"]
        if images.shape != (len(rows), 2, size, size):
            raise ValueError("Image array shape mismatch")
        for start in range(0, len(rows), 32):
            block = images[start : start + 32]
            if not np.isfinite(block).all() or block.min() < 0:
                raise ValueError("Invalid image pixels")
            np.testing.assert_allclose(block.mean((2, 3)), 1, atol=2e-6)
        noise = np.load(folder / "counts.npy", mmap_mode="r")
        if not np.isfinite(noise).all() or noise.min() <= 0:
            raise ValueError("Invalid Poisson scales")
        family_counts = collections.Counter(row["family"] for row in rows)
        if set(family_counts) - set(info["families"]):
            raise ValueError("Unexpected band family in split")
        # Replay one example per family from its configuration seed.
        replayed = set()
        for i, row in enumerate(rows):
            if row["family"] not in replayed:
                replayed.add(row["family"])
                regenerated, grids, sigma, bands, _ = simulate(
                    row["seed"], row["family"], size
                )
                np.testing.assert_array_equal(regenerated, images[i])
                for name, array in [
                    ("grids.npy", grids),
                    ("self_energy.npy", sigma),
                    ("bands.npy", bands),
                ]:
                    np.testing.assert_array_equal(
                        array, np.load(folder / name, mmap_mode="r")[i]
                    )
        report[split] = {
            "count": len(rows),
            "families": dict(family_counts),
            "matrix_modes": dict(
                collections.Counter(row["matrix_element"]["mode"] for row in rows)
            ),
            "minimum_visible_band_column_fraction": min(
                row["visible_band_column_fraction"] for row in rows
            ),
            "maximum_bare_second_moment": max(
                row["bare_second_moment"] for row in rows
            ),
            "maximum_retries": max(row["rejection_attempts"] for row in rows),
            "replayed_families": sorted(replayed),
        }
    if set(report["train"]["families"]) & set(report["ood"]["families"]):
        raise ValueError("Held-out family leakage")
    return {"passed": True, "total_configurations": len(seen), "splits": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    report = audit(args.corpus)
    (args.corpus / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
