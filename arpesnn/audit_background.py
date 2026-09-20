"""Audit background corpus arrays, seed/family isolation and signal regeneration."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from background_data import signal_job
from band_models import HELD_OUT_FAMILIES


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "additive_background_v1":
        raise ValueError("Wrong schema")
    seen = set()
    report = {}
    for split, info in manifest["splits"].items():
        folder = root / split
        if (
            hashlib.sha256((folder / "parameters.jsonl").read_bytes()).hexdigest()
            != info["parameters_sha256"]
        ):
            raise ValueError("Metadata checksum mismatch")
        rows = [
            json.loads(s)
            for s in (folder / "parameters.jsonl").read_text().splitlines()
        ]
        signal = np.load(folder / "signal.npy", mmap_mode="r")
        grids = np.load(folder / "grids.npy", mmap_mode="r")
        assert signal.shape == (info["count"], manifest["size"], manifest["size"])
        assert grids.shape == (info["count"], 3, manifest["size"])
        assert len(rows) == info["count"]
        for start in range(0, len(rows), 64):
            chunk = signal[start : start + 64]
            assert np.isfinite(chunk).all() and chunk.min() >= 0
            np.testing.assert_allclose(chunk.mean((1, 2)), 1, atol=2e-6)
        sampled = set()
        for i, row in enumerate(rows):
            assert row["seed"] == info["first_seed"] + i and row["seed"] not in seen
            seen.add(row["seed"])
            assert row["background_fraction"] == 0
            assert (row["family"] in HELD_OUT_FAMILIES) == (split == "ood")
            if row["family"] not in sampled:
                a, g, metadata = signal_job(
                    (row["seed"], row["family"], manifest["size"])
                )
                np.testing.assert_array_equal(signal[i], a)
                np.testing.assert_array_equal(grids[i], g)
                assert metadata == row
                sampled.add(row["family"])
        report[split] = {
            "count": len(rows),
            "replayed_families": len(sampled),
            "broad_flat": sum(
                r["retained_signal_feature"] == "broad_flat" for r in rows
            ),
            "weak_band": sum(r["retained_signal_feature"] == "weak_band" for r in rows),
        }
    (root / "audit.json").write_text(
        json.dumps({"passed": True, "splits": report}, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("corpus", type=Path)
    audit(p.parse_args().corpus)
