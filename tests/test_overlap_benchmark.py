"""Checks that fitted errors measure processing/noise, not a broken fitter."""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arpesnn"))
from benchmark_overlap import synth, fit, analyze_job
from benchmark_denoising import filtered


class OverlapBenchmarkTests(unittest.TestCase):
    def test_noiseless_overlapping_components_are_recovered(self):
        for case in (
            "single",
            "double_0.35",
            "weak_double",
            "avoided_crossing",
            "broad_incoherent",
        ):
            with self.subTest(case=case):
                clean, centers, widths, _ = synth(case)
                result = fit(clean[:, 128], len(centers))
                order = np.argsort(centers[:, 128])
                self.assertTrue(result["success"])
                np.testing.assert_allclose(
                    result["centers"], centers[order, 128], atol=1e-5
                )
                np.testing.assert_allclose(
                    result["gamma"], widths[order, 128], atol=1e-5
                )

    def test_overlapping_peak_maximum_is_not_a_component_center(self):
        from benchmark_overlap import E

        clean, centers, _, _ = synth("double_0.35")
        maximum = E[clean[:, 128].argmax()]
        self.assertGreater(np.min(abs(centers[:, 128] - maximum)), 0.003)

    def test_filter_boundary_constant_preservation(self):
        data = np.full((256, 256), 3.0, dtype=np.float32)
        for c in [("gaussian", 2.0, 1.0, 0), ("fourier", 0.1, 0.2, 4)]:
            np.testing.assert_allclose(filtered(data, c), data, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
