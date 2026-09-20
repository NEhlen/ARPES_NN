import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arpesnn"))
from band_models import (
    FAMILIES,
    TRAIN_FAMILIES,
    HELD_OUT_FAMILIES,
    sample_parameters,
    band_energies,
)
from corpus import simulate, spectral_image, causal_self_energy, create_corpus
from compact_model import CompactUNet
from train_tasks import CorpusDataset, ridge_error
from dispersion_relations import Dispersion
from spectral_function import SpectralFunction, SelfEnergy
from apply_model import load_model


class PhysicsTests(unittest.TestCase):
    def test_legacy_vectorization_matches_reference(self):
        def bands(k, params):
            return np.array([-0.5 + k[0] ** 2, -0.2 - k[0] ** 2])

        spectral = SpectralFunction(
            Dispersion({}, bands),
            SelfEnergy("self-energy", {"ai": 0.02, "bi": 0.01, "ar": 0.05}),
        )
        shape = (37, 29)
        energy = -1 + np.arange(shape[0]) * 1.2 / shape[0]
        k = np.array([[-0.8 + i * 1.6 / shape[1], 0] for i in range(shape[1])])
        reference = np.array(
            [
                [
                    spectral._spectral(
                        q, float(e), spectral.self_energy, spectral.dispersion
                    )
                    for q in k
                ]
                for e in energy
            ]
        )
        reference -= reference.min()
        reference /= reference.mean()
        actual = spectral.generate_base([-0.8, 0], [0.8, 0], -1, 0.2, shape).spectrum
        np.testing.assert_allclose(actual, reference, atol=1e-12)

    def test_all_families_reproducible_finite_and_offcenter(self):
        centers, angles, offsets, modes = [], [], [], set()
        for i, family in enumerate(FAMILIES):
            images, grids, sigma, bands, meta = simulate(900 + i, family, 32)
            np.testing.assert_array_equal(images, simulate(900 + i, family, 32)[0])
            self.assertEqual(images.shape, (2, 32, 32))
            self.assertTrue(np.isfinite(images).all())
            self.assertGreaterEqual(images.min(), 0)
            self.assertGreaterEqual(meta["visible_band_column_fraction"], 0.2)
            self.assertLessEqual(meta["bare_second_moment"], 150)
            self.assertEqual(meta["momentum_oversampling"], 4)
            np.testing.assert_allclose(images.mean((1, 2)), 1, atol=1e-6)
            self.assertTrue((sigma[1] < 0).all())
            self.assertTrue(np.isfinite(grids).all())
            self.assertTrue(
                np.isfinite(
                    bands[
                        : (
                            meta["bands"]["band_count"]
                            if family == "multiband_parabolic"
                            else 1
                        )
                    ]
                ).all()
            )
            w = meta["window"]
            centers.append((w["smin"] + w["smax"]) / 2)
            angles.append(w["cut_angle_rad"])
            offsets.append(w["perpendicular_inverse_angstrom"])
            modes.add(meta["matrix_element"]["mode"])
        self.assertGreater(np.std(centers), 0.4)
        self.assertGreater(np.std(angles), 1)
        self.assertTrue(any(abs(v) > 0.2 for v in offsets))
        self.assertEqual(modes, {"constant", "symmetric", "asymmetric"})
        self.assertFalse(set(TRAIN_FAMILIES) & set(HELD_OUT_FAMILIES))

    def test_periodicity_gap_and_causality(self):
        p = sample_parameters(np.random.default_rng(5), "square_lattice")
        k = np.linspace(-2, 2, 101)
        np.testing.assert_allclose(
            band_energies(k, np.zeros_like(k), p),
            band_energies(k + 2 * np.pi / p["lattice_a"], np.zeros_like(k), p),
            atol=1e-12,
        )
        p["family"] = "avoided_crossing"
        # Exact diagonal crossing at ky=0, v*k = -0.6*v*k + split.
        crossing = p["split"] / (1.6 * p["velocity"])
        e = band_energies(np.array([crossing]), np.array([0.0]), p)
        self.assertAlmostEqual(float(e[1, 0] - e[0, 0]), 2 * p["coupling"])
        se = {
            "gamma": 0.01,
            "centers": [-0.2, 0.2],
            "widths": [0.05, 0.05],
            "weights": [0.02, 0.02],
        }
        sigma = causal_self_energy(np.array([-1.0, 0.0, 1.0]), se)
        self.assertAlmostEqual(sigma[1].real, 0)
        self.assertTrue((sigma.imag < 0).all())

    def test_fast_spectrum_matches_complex_formula(self):
        energy = np.linspace(-1, 0.2, 80)
        bands = np.array([np.linspace(-0.7, -0.2, 64), np.linspace(-0.2, -0.6, 64)])
        sigma = np.full(80, 0.03 - 0.02j)
        reference = (
            -np.imag(
                1 / (energy[:, None, None] - bands[None, :, :] - sigma[:, None, None])
            )
            / np.pi
        ).sum(1)
        np.testing.assert_allclose(
            spectral_image(energy, bands, sigma), reference, rtol=1e-12
        )

    def test_ridge_metric_uses_physical_energy_and_visible_bands(self):
        energies = np.array([[-1.0, -0.5, 0.0]])
        bands = np.array([[[-0.5, -1.0], [np.nan, np.nan]]])
        prediction = np.array([[[0.0, 5.0], [5.0, 0.0], [0.0, 0.0]]])
        error, count = ridge_error(prediction, energies, bands)
        self.assertEqual(error, 0)
        self.assertEqual(count, 2)
        error, _ = ridge_error(prediction[:, ::-1], energies, bands)
        self.assertEqual(error, 1)

    def test_compact_checkpoint_roundtrip_and_identity(self):
        torch.set_num_threads(2)
        x = torch.rand(2, 1, 32, 32)
        for task in ("denoise", "bareband"):
            m = CompactUNet(task).eval()
            with torch.no_grad():
                y = m(x)
            self.assertEqual(y.shape, x.shape)
            if task == "denoise":
                torch.testing.assert_close(y, x)
            self.assertTrue((y >= 0).all())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                torch.save(
                    {
                        "architecture": "compact_unet_v1",
                        "task": task,
                        "base_channels": 16,
                        "state_dict": m.state_dict(),
                    },
                    path,
                )
                loaded, channels = load_model(path, torch.device("cpu"))
                self.assertEqual(channels, 1)
                with torch.no_grad():
                    torch.testing.assert_close(loaded(x), y)


class CorpusTests(unittest.TestCase):
    def test_split_isolation_fresh_noise_and_validation_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corpus"
            create_corpus(
                root,
                {"train": 10, "validation": 3, "test": 3, "ood": 2},
                size=16,
                workers=1,
            )
            train = CorpusDataset(root, "train")
            val = CorpusDataset(root, "validation")
            test = CorpusDataset(root, "test")
            ood = CorpusDataset(root, "ood")
            self.assertEqual(
                len({d.info["first_seed"] for d in [train, val, test, ood]}), 4
            )
            x, y, b, _ = train[0]
            train.epoch.value = 1
            self.assertFalse(torch.equal(x, train[0][0]))
            self.assertTrue(torch.equal(b, train[0][2]))
            fixed = val[0]
            val.epoch.value = 50
            self.assertTrue(torch.equal(fixed[0], val[0][0]))
            np.testing.assert_allclose(fixed[0].numpy().mean(), 1, atol=1e-6)
            self.assertFalse(set(ood.info["families"]) & set(train.info["families"]))
            with self.assertRaises(FileExistsError):
                create_corpus(root, {"train": 10}, size=16, workers=1)


if __name__ == "__main__":
    unittest.main()
