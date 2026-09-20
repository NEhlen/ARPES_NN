import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arpesnn"))
from background_data import (
    make_background,
    signal_job,
    BackgroundDataset,
    make_background_v2,
    V2_REGIMES,
    local_band_mask,
)
from background_model import BackgroundNet, load_background
from corpus import simulate
from apply_background import apply


class BackgroundTests(unittest.TestCase):
    def test_background_switch_preserves_band_physics_and_default(self):
        old = simulate(12, "parabolic", 32)
        explicit = simulate(12, "parabolic", 32, include_background=True)
        signal = simulate(12, "parabolic", 32, include_background=False)
        for a, b in zip(old[:4], explicit[:4]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(old[1], signal[1])
        np.testing.assert_array_equal(old[2], signal[2])
        np.testing.assert_array_equal(old[3], signal[3])
        self.assertEqual(signal[4]["background_fraction"], 0)
        self.assertGreater(np.max(abs(signal[0][0] - old[0][0])), 0)

    def test_background_labels_and_zero_controls(self):
        signal, _, meta = signal_job((123, "parabolic", 32))
        self.assertEqual(meta["background_fraction"], 0)
        for kind in [
            "none",
            "constant",
            "tilted",
            "hump",
            "step",
            "correlated",
            "mixture",
        ]:
            b, m = make_background(signal, np.random.default_rng(3), kind)
            self.assertTrue(np.isfinite(b).all())
            self.assertGreaterEqual(b.min(), 0)
            self.assertAlmostEqual(
                float(b.mean() / signal.mean()),
                m["background_to_signal_mean"],
                places=5,
            )
            if kind == "none":
                np.testing.assert_array_equal(b, np.zeros_like(signal))

    def test_v2_contrast_targets(self):
        signal, _, _ = signal_job((123, "parabolic", 32))
        for regime in V2_REGIMES:
            b, meta = make_background_v2(signal, np.random.default_rng(9), regime)
            self.assertTrue(np.isfinite(b).all())
            self.assertGreaterEqual(float(b.min()), 0)
            if regime == "zero":
                self.assertEqual(float(b.max()), 0)
            elif regime.startswith("mean_"):
                lo, hi = {
                    "mean_weak": (0.05, 0.5),
                    "mean_comparable": (0.5, 2),
                    "mean_dominant": (2, 10),
                }[regime]
                self.assertTrue(lo <= b.mean() / signal.mean() <= hi)
            else:
                lo, hi = {
                    "local_high": (2, 10),
                    "local_comparable": (0.5, 2),
                    "local_buried": (0.05, 0.5),
                }[regime]
                mask = local_band_mask(signal)
                self.assertTrue(lo <= np.median(signal[mask] / b[mask]) <= hi)

    def test_fresh_training_but_fixed_test_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"schema": "additive_background_v1", "size": 32, "splits": {}}
            for split in ["train", "test"]:
                (root / split).mkdir()
                np.save(
                    root / split / "signal.npy", np.ones((1, 32, 32), dtype="float32")
                )
                manifest["splits"][split] = {
                    "count": 1,
                    "first_seed": 100 if split == "train" else 200,
                }
            (root / "manifest.json").write_text(json.dumps(manifest))
            for split, profile in [
                (s, p) for s in ["train", "test"] for p in ["v1", "v2"]
            ]:
                ds = BackgroundDataset(root, split, profile=profile)
                a = ds[0]["input"].numpy()
                ds.epoch = 1
                b = ds[0]["input"].numpy()
                self.assertEqual(np.array_equal(a, b), split == "test")

    def test_signed_subtraction_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        model = BackgroundNet().eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "model.pt"
            torch.save(
                {
                    "architecture": "background_coarse_v1",
                    "task": "background",
                    "base_channels": 16,
                    "input_shape": [32, 32],
                    "state_dict": model.state_dict(),
                },
                p,
            )
            loaded, _ = load_background(p, torch.device("cpu"))
            raw = np.ones((32, 32), dtype="float32")
            raw[:3] = 0
            np.save(root / "input.npy", raw)
            apply(loaded, root / "input.npy", root / "output", torch.device("cpu"), 32)
            data = np.load(root / "output/decomposition.npz")
            np.testing.assert_allclose(
                data["corrected"] + data["background"], data["model_input"], atol=1e-6
            )
            self.assertLess(data["corrected"].min(), 0)
            with torch.inference_mode():
                np.testing.assert_allclose(
                    model(torch.ones(1, 1, 32, 32)).numpy(),
                    loaded(torch.ones(1, 1, 32, 32)).numpy(),
                )


if __name__ == "__main__":
    unittest.main()
