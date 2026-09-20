"""Reproducible legacy speed/accuracy benchmark and v2 throughput measurement."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from band_models import FAMILIES
from corpus import simulate
from dispersion_relations import Dispersion
from spectral_function import SpectralFunction, SelfEnergy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arpesnn/models/preflight_v2/generation_benchmark.json"),
    )
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    sf = SpectralFunction(
        Dispersion({}, lambda k, p: np.array([-0.5 + 2 * k[0] ** 2, -0.7 - k[0] ** 2])),
        SelfEnergy("self-energy", {"ai": 0.02, "bi": 0.01, "ar": 0.03}),
    )
    k = np.array([[-1 + i * 2 / 256, 0] for i in range(256)])
    energies = np.arange(256) * 1.2 / 256 - 1
    started = time.perf_counter()
    reference = np.array(
        [
            [sf._spectral(q, float(e), sf.self_energy, sf.dispersion) for q in k]
            for e in energies
        ]
    )
    reference -= reference.min()
    reference /= reference.mean()
    old_seconds = time.perf_counter() - started
    timings = []
    for _ in range(10):
        started = time.perf_counter()
        image = sf.generate_base([-1, 0], [1, 0], -1, 0.2, (256, 256)).spectrum
        timings.append(time.perf_counter() - started)
    np.testing.assert_allclose(image, reference, rtol=1e-11, atol=1e-12)
    started = time.perf_counter()
    for i in range(args.samples):
        simulate(20000 + i, FAMILIES[i % len(FAMILIES)])
    report = {
        "legacy_reference_seconds": old_seconds,
        "vectorized_median_seconds": float(np.median(timings)),
        "speedup": old_seconds / np.median(timings),
        "max_absolute_difference": float(np.max(abs(image - reference))),
        "v2_mean_seconds_per_configuration": (time.perf_counter() - started)
        / args.samples,
        "v2_pixel_integration": [4, 4],
        "v2_samples_benchmarked": args.samples,
        "shape": [256, 256],
        "legacy_reference_bands": 2,
    }
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
