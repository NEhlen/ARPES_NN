import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arpesnn"))
from generate_tasks import generate_sample, spectral_image
from generate_data import save_training_sample
from nn_pytorch import load_data
from apply_model import build_model_input, load_experimental_file
from sp2 import load_sp2


class TaskGenerationTests(unittest.TestCase):
    def test_reproducible_pairs_and_physical_truth(self):
        noisy, targets, params, truth = generate_sample(19, 32)
        repeat, repeated_targets, _, _ = generate_sample(19, 32)
        np.testing.assert_array_equal(noisy, repeat)
        np.testing.assert_array_equal(targets["denoise"], repeated_targets["denoise"])
        self.assertEqual(noisy.shape, (32, 32))
        self.assertAlmostEqual(float(noisy.mean()), 1.0, places=6)
        for target in targets.values():
            self.assertTrue(np.isfinite(target).all())
            self.assertGreaterEqual(target.min(), 0)
        self.assertFalse(np.array_equal(targets["denoise"], targets["bareband"]))
        # Verify denoising target is the expectation of the Poisson input,
        # within statistical error across all independent pixels.
        counts = params["simulation_data"]["counts_per_mean_pixel"]
        scale = params["simulation_data"]["input_normalization_scale"]
        expected = targets["denoise"] * scale * counts
        observed = noisy * scale * counts
        z = (observed - expected).sum() / np.sqrt(expected.sum())
        self.assertLess(abs(z), 6)
        # Stored bare truth reconstructs the target, with zero real Sigma.
        from scipy.special import expit
        from physics_parameters import _kb

        energies = truth["energy_eV"]
        gamma = params["simulation_data"]["bare_lorentzian_hwhm_eV"]
        bare = spectral_image(
            energies, truth["bare_dispersion_eV"], np.full(32, -1j * gamma)
        )
        bare *= expit(-energies / (_kb * params["simulation_data"]["temperature_K"]))[
            :, None
        ]
        bare /= bare.mean()
        np.testing.assert_allclose(bare, targets["bareband"], rtol=1e-6)

    def test_spectral_peak_recovers_bare_energy(self):
        energy = np.linspace(-1, 0, 1001)
        bands = np.array([[-0.6, -0.2]])
        bare = spectral_image(energy, bands, np.full(1001, -0.01j))
        shifted = spectral_image(energy, bands, np.full(1001, 0.1 - 0.01j))
        np.testing.assert_allclose(energy[bare.argmax(axis=0)], bands[0], atol=0.001)
        np.testing.assert_allclose(
            energy[shifted.argmax(axis=0)], bands[0] + 0.1, atol=0.001
        )

    def test_task_mixing_and_legacy_rejected(self):
        noisy, targets, params, _ = generate_sample(3, 16)
        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "001" / "sample")
            save_training_sample(
                prefix,
                noisy,
                targets["denoise"],
                dict(params, task="denoise"),
                "npy",
                False,
            )
            inputs, outputs = load_data(tmp, expected_task="denoise")
            self.assertEqual(inputs.shape, (1, 16, 16))
            with self.assertRaisesRegex(ValueError, "task mismatch"):
                load_data(tmp, expected_task="bareband")
            save_training_sample(
                str(Path(tmp) / "002" / "sample"),
                noisy,
                targets["bareband"],
                params,
                "npy",
                False,
            )
            with self.assertRaisesRegex(ValueError, "task mismatch"):
                load_data(tmp)


class SP2Tests(unittest.TestCase):
    def fixture(self, directory, payload="0 1 2 3 4 5"):
        path = Path(directory) / "spectrum.sp2"
        path.write_text(
            'P2\n# Images = "Corrected Raw"\n# ERange = 20 22 # eV\n# aRange = -5 5 # deg\n# aUnit = "deg"\n3 2 9\n'
            + payload
            + "\nP2\n3 2 9\n9 8 7 6 5 4\n"
        )
        return path

    def test_multiple_blocks_orientation_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(tmp)
            blocks = load_sp2(path)
            self.assertEqual(len(blocks), 2)
            np.testing.assert_array_equal(blocks[0].intensity, [[0, 3], [1, 4], [2, 5]])
            np.testing.assert_array_equal(blocks[0].energy_eV, [20, 21, 22])
            np.testing.assert_array_equal(blocks[0].angle_deg, [-5, 5])
            np.testing.assert_array_equal(blocks[1].intensity, [[9, 6], [8, 5], [7, 4]])
            self.assertIsNone(blocks[1].energy_eV)
            with self.assertRaisesRegex(ValueError, "instrument correction"):
                load_experimental_file(path, 1)
            data, meta = load_experimental_file(path)
            with self.assertRaisesRegex(ValueError, "Calibrate EF"):
                build_model_input(data, meta, (16, 16), 3)
            self.assertEqual(build_model_input(data, meta, (16, 16), 1).shape, (16, 16))

    def test_truncated_and_invalid_payloads_rejected(self):
        for payload in ("0 1", "0 1 2 3 4 nan", "0 1 2 3 4 -1", "0 1 2 3 4 10"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    load_sp2(self.fixture(tmp, payload))

    def test_local_experimental_files(self):
        files = sorted(
            (
                Path(__file__).resolve().parents[1]
                / "data/experiment_data/Elettra-Feb18/09"
            ).glob("*.sp2")
        )
        if not files:
            self.skipTest("Local experimental data not installed")
        for path in files:
            with self.subTest(path=path.name):
                blocks = load_sp2(path)
                self.assertEqual(len(blocks), 2)
                self.assertEqual(blocks[0].intensity.shape, (640, 480))
                self.assertTrue(np.isfinite(blocks[0].intensity).all())


if __name__ == "__main__":
    unittest.main()
