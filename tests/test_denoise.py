import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arpesnn"))
from compact_model import CompactUNet
from denoise import predict, load_denoiser
from apply_model import resize_array


class DenoiseTests(unittest.TestCase):
    def test_identity_checkpoint_preserves_resampled_intensity_units(self):
        torch.set_num_threads(1)
        data = np.random.default_rng(0).uniform(20, 200, (48, 80)).astype("float32")
        x, y, scale = predict(
            data, CompactUNet("denoise").eval(), torch.device("cpu"), 32
        )
        np.testing.assert_allclose(x, resize_array(data, (32, 32)))
        np.testing.assert_allclose(y, x, rtol=1e-6)
        self.assertAlmostEqual(scale, float(x.mean()))

    def test_invalid_inputs_and_bareband_checkpoint_rejected(self):
        model = CompactUNet("denoise").eval()
        for data in [
            np.zeros((32, 32)),
            np.ones((32, 32)) * -1,
            np.full((32, 32), np.nan),
            np.ones(32),
        ]:
            with self.assertRaises(ValueError):
                predict(data, model, torch.device("cpu"), 32)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.pt"
            torch.save({"architecture": "compact_unet_v1", "task": "bareband"}, p)
            with self.assertRaises(ValueError):
                load_denoiser(p, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
