"""Generate paired denoising and bare-band datasets without changing legacy data."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from generate_data import (
    random_band_parameters,
    random_self_energy,
    save_training_sample,
)
from physics_parameters import _kb
from spectral_function import SelfEnergy

TASKS = ("denoise", "bareband")


def spectral_image(energies, bands, sigma):
    """A(E,k), summed over bands; energy and dispersion use the same grid."""
    return (
        -np.imag(
            1 / (energies[:, None, None] - bands[None, :, :] - sigma[:, None, None])
        )
        / np.pi
    ).sum(axis=1)


def generate_sample(seed: int, size: int = 256):
    if size < 16 or size % 4:
        raise ValueError("size must be at least 16 and divisible by 4")
    # Existing family samplers use NumPy's global RNG. Restore it for callers.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        family, band_fn, band_params = random_band_parameters()
        se_params = random_self_energy(kink=True)
    finally:
        np.random.set_state(state)
    rng = np.random.default_rng(seed)
    emin = float(rng.uniform(-3.5, -0.4))
    emax = 0.15
    band_params["energy_shift"] = float(rng.uniform(0.65 * emin, -0.05))
    kspan = float(rng.uniform(0.7, 2.0))
    kcenter = band_params["k0"]
    energies = np.linspace(emin, emax, size)
    momenta = np.linspace(kcenter - kspan / 2, kcenter + kspan / 2, size)
    bands = band_fn(np.stack([momenta, np.zeros(size)]), band_params)
    sigma = np.array([SelfEnergy("self-energy", se_params)(float(e)) for e in energies])
    temperature = float(rng.uniform(10, 180))
    occupation = expit(-energies / (_kb * temperature))[:, None]
    intrinsic = spectral_image(energies, bands, sigma)
    # Smooth positive matrix element and background are part of the clean measurement.
    x = np.linspace(-1, 1, size)
    envelope = (
        0.3
        + 0.7 * np.exp(-(((x - rng.uniform(-0.5, 0.5)) / rng.uniform(0.4, 1.4)) ** 2))
    )[None, :]
    clean = intrinsic * occupation * envelope
    clean += float(rng.uniform(0.01, 0.15)) * clean.mean() * (1 + 0.2 * x[:, None])
    energy_sigma = float(rng.uniform(0.004, 0.04))
    momentum_sigma = float(rng.uniform(0.002, 0.015))
    clean = gaussian_filter(
        clean,
        (
            energy_sigma / (energies[1] - energies[0]),
            momentum_sigma / (momenta[1] - momenta[0]),
        ),
        mode="nearest",
    )
    clean /= clean.mean()
    counts_per_mean_pixel = float(10 ** rng.uniform(0.5, 3.0))
    noisy = rng.poisson(clean * counts_per_mean_pixel) / counts_per_mean_pixel
    scale = noisy.mean()
    if scale <= 0 or not np.isfinite(clean).all():
        raise ValueError("Invalid generated spectrum")
    # Match inference normalization, with exactly the same scale for input/clean.
    noisy /= scale
    clean /= scale
    # Finite-grid proxy for delta-function bare bands; no real self-energy.
    gamma = max(0.005, float(energies[1] - energies[0]))
    bare = spectral_image(energies, bands, np.full(size, -1j * gamma)) * occupation
    bare /= bare.mean()
    params = {
        "schema_version": 1,
        "sample_seed": seed,
        "family": family,
        "dispersion_params": band_params,
        "selfenergy_params": se_params,
        "spectrum_params": {
            "Emin": emin,
            "Emax": emax,
            "kmin": [float(momenta[0]), 0.0],
            "kmax": [float(momenta[-1]), 0.0],
            "shape": [size, size],
        },
        "simulation_data": {
            "temperature_K": temperature,
            "energy_sigma_eV": energy_sigma,
            "momentum_sigma_inverse_angstrom": momentum_sigma,
            "counts_per_mean_pixel": counts_per_mean_pixel,
            "input_normalization_scale": float(scale),
            "bare_lorentzian_hwhm_eV": gamma,
        },
        "energy_convention": "E_minus_EF_eV",
        "momentum_units": "inverse_angstrom",
        "axis_order": "energy,momentum",
    }
    truth = {
        "energy_eV": energies,
        "momentum_inverse_angstrom": momenta,
        "bare_dispersion_eV": bands,
        "self_energy_real_eV": sigma.real,
        "self_energy_imag_eV": sigma.imag,
    }
    return (
        noisy.astype(np.float32),
        {"denoise": clean.astype(np.float32), "bareband": bare.astype(np.float32)},
        params,
        truth,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "both"), default="both")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.getenv("DATASET_PATH", "arpesnn/dataset")),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First sample seed; use a new range to append.",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--coordinate-input",
        action="store_true",
        help="Add physical E,k channels; otherwise intensity only.",
    )
    args = parser.parse_args()
    tasks = TASKS if args.task == "both" else (args.task,)
    if args.samples < 1 or args.seed < 0 or args.seed + args.samples > 2**32:
        parser.error("samples must be positive and seeds must fit uint32")
    if args.size < 16 or args.size % 4:
        parser.error("size must be at least 16 and divisible by 4")
    for task in tasks:
        for seed in range(args.seed, args.seed + args.samples):
            if (args.output_root / task / f"seed-{seed:09d}").exists():
                parser.error(
                    f"Sample {seed} already exists for {task}; choose a new --seed range."
                )
    for seed in range(args.seed, args.seed + args.samples):
        noisy, targets, params, truth = generate_sample(seed, args.size)
        for task in tasks:
            prefix = args.output_root / task / f"seed-{seed:09d}" / "spectrum"
            metadata = dict(
                params, task=task, input_channels=3 if args.coordinate_input else 1
            )
            save_training_sample(
                str(prefix),
                noisy,
                targets[task],
                metadata,
                "npy",
                args.coordinate_input,
            )
            np.savez(prefix.parent / "physical_truth.npz", **truth)
        print(
            f"seed={seed} family={params['family']} tasks={','.join(tasks)}", flush=True
        )


if __name__ == "__main__":
    main()
