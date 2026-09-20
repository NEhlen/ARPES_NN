"""Fast v2 simulation and disk-backed corpus; noise is sampled by the trainer."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from band_models import (
    FAMILIES,
    TRAIN_FAMILIES,
    HELD_OUT_FAMILIES,
    band_energies,
    sample_parameters,
)

SCHEMA_VERSION = 2
KB = 8.617333262e-5


def spectral_image(energy, bands, sigma):
    """Lorentzian spectral sum without per-pixel loops or a complex 3D tensor."""
    gamma = -np.imag(sigma)[:, None]
    center = energy[:, None] - np.real(sigma)[:, None]
    result = np.zeros((len(energy), bands.shape[1]), dtype=np.float64)
    for band in bands:
        result += gamma / ((center - band[None, :]) ** 2 + gamma**2)
    return result / np.pi


def causal_self_energy(energy, p):
    """Retarded pole model: Im Sigma <= 0, paired real/imaginary parts.

    Phenomenological causal resonances, not a microscopic phonon calculation.
    A real constant aligns Re Sigma(0)=0. Each pole lies below the real axis.
    """
    result = np.full(energy.shape, -1j * p["gamma"], dtype=np.complex128)
    at_zero = 0j
    for center, width, weight in zip(p["centers"], p["widths"], p["weights"]):
        result += weight / (energy - center + 1j * width)
        at_zero += weight / (-center + 1j * width)
    return result - at_zero.real


def _simulate_once(seed, family, size=256, include_background=True):
    rng = np.random.default_rng(seed)
    p = sample_parameters(rng, family)
    span = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    center = float(rng.uniform(-2.5, 2.5))
    theta = float(rng.uniform(0, 2 * np.pi))
    perpendicular = 0.0 if rng.random() < 0.25 else float(rng.uniform(-1.2, 1.2))
    s = np.linspace(center - span / 2, center + span / 2, size)
    kx = s * np.cos(theta) - perpendicular * np.sin(theta)
    ky = s * np.sin(theta) + perpendicular * np.cos(theta)
    bands = band_energies(kx, ky, p)
    ewidth = float(np.exp(rng.uniform(np.log(0.35), np.log(4.0))))
    emax = float(
        rng.uniform(0.03, 0.25) if rng.random() < 0.35 else rng.uniform(-2.5, -0.03)
    )
    emin = emax - ewidth
    energy = np.linspace(emin, emax, size)
    # Place one branch inside the window at a random point on the cut. This
    # avoids a corpus dominated by empty images without centering every band.
    anchor_band = int(rng.integers(len(bands)))
    anchor_pixel = int(rng.integers(size // 8, 7 * size // 8))
    desired = float(rng.uniform(emin + 0.15 * ewidth, min(emax - 0.15 * ewidth, -0.01)))
    p["shift"] = desired - float(bands[anchor_band, anchor_pixel])
    bands += p["shift"]
    n_modes = int(rng.integers(1, 4))
    omega = rng.uniform(0.04, 0.7, n_modes)
    widths = rng.uniform(0.02, 0.25, n_modes)
    weights = rng.uniform(0.015, 0.25, n_modes) * omega
    se = {
        "kind": "causal_poles",
        "gamma": float(rng.uniform(0.003, 0.03)),
        "centers": np.concatenate([-omega, omega]).tolist(),
        "widths": np.tile(widths, 2).tolist(),
        "weights": np.tile(weights, 2).tolist(),
    }
    sigma = causal_self_energy(energy, se)
    temperature = float(rng.uniform(10, 200))
    # Integrate a 4 x 4 subpixel grid before returning detector pixels.
    # Both axes matter: steep narrow bands otherwise become dotted lines.
    de, dk = energy[1] - energy[0], s[1] - s[0]
    oversampling = 4
    fine_energy = np.linspace(
        emin - de / 2 + de / (2 * oversampling),
        emax + de / 2 - de / (2 * oversampling),
        size * oversampling,
    )
    fine_s = np.linspace(
        s[0] - dk / 2 + dk / (2 * oversampling),
        s[-1] + dk / 2 - dk / (2 * oversampling),
        size * oversampling,
    )
    fine_bands = band_energies(
        fine_s * np.cos(theta) - perpendicular * np.sin(theta),
        fine_s * np.sin(theta) + perpendicular * np.cos(theta),
        p,
    )
    occupation = expit(-fine_energy / (KB * temperature))[:, None]
    intrinsic = spectral_image(
        fine_energy, fine_bands, causal_self_energy(fine_energy, se)
    )
    x = np.linspace(-1, 1, size * oversampling)
    mode = str(rng.choice(["constant", "symmetric", "asymmetric"]))
    envelope_center = float(rng.uniform(-0.8, 0.8)) if mode == "asymmetric" else 0.0
    envelope_width = float(rng.uniform(0.3, 1.8))
    envelope = (
        np.ones((1, size * oversampling))
        if mode == "constant"
        else (0.15 + 0.85 * np.exp(-(((x - envelope_center) / envelope_width) ** 2)))[
            None, :
        ]
    )
    clean = intrinsic * occupation * envelope
    # Background slope is along energy, not momentum.
    background = float(rng.uniform(0, 0.12))
    if not include_background:
        background = 0.0
    clean += (
        background
        * clean.mean()
        * (1 + 0.3 * np.linspace(-1, 1, size * oversampling)[:, None])
    )
    energy_sigma = float(rng.uniform(0.003, min(0.05, ewidth / 10)))
    momentum_sigma = float(rng.uniform(0.001, min(0.015, span / 10)))
    clean = gaussian_filter(
        clean,
        (energy_sigma * oversampling / de, momentum_sigma * oversampling / dk),
        mode="nearest",
    )
    # A small fraction have cropped detector acceptance; the denoiser preserves it.
    acceptance = np.ones(size)
    if rng.random() < 0.2:
        edge = int(rng.integers(1, max(2, size // 15)))
        acceptance[:edge] = 0
    if rng.random() < 0.2:
        edge = int(rng.integers(1, max(2, size // 15)))
        acceptance[-edge:] = 0
    clean = clean.reshape(size, oversampling, size, oversampling).mean(axis=(1, 3))
    clean *= acceptance[None, :]
    clean /= clean.mean()
    gamma = max(0.003, float(de))
    bare = (
        spectral_image(
            fine_energy, fine_bands, np.full(size * oversampling, -1j * gamma)
        )
        * occupation
    )
    bare = bare.reshape(size, oversampling, size, oversampling).mean(axis=(1, 3))
    bare /= bare.mean()
    counts = float(10 ** rng.uniform(0.5, 3.3))
    images = np.stack([clean, bare]).astype(np.float32)
    if not np.isfinite(images).all() or images.min() < 0:
        raise ValueError(f"Nonfinite simulation for seed {seed}")
    truth_bands = np.full((4, size), np.nan, dtype=np.float32)
    truth_bands[: len(bands)] = bands
    meta = {
        "seed": seed,
        "family": family,
        "bands": p,
        "self_energy": se,
        "window": {
            "emin": emin,
            "emax": emax,
            "smin": float(s[0]),
            "smax": float(s[-1]),
            "cut_angle_rad": theta,
            "perpendicular_inverse_angstrom": perpendicular,
        },
        "matrix_element": {
            "mode": mode,
            "center_normalized": envelope_center,
            "width_normalized": envelope_width,
        },
        "temperature_K": temperature,
        "energy_sigma_eV": energy_sigma,
        "momentum_sigma_inverse_angstrom": momentum_sigma,
        "background_fraction": background,
        "bare_hwhm_eV": gamma,
        "energy_oversampling": oversampling,
        "momentum_oversampling": oversampling,
        "counts_per_mean_pixel": counts,
        "anchor_band": anchor_band,
        "anchor_pixel": anchor_pixel,
    }
    return (
        images,
        np.stack([energy, kx, ky]).astype(np.float32),
        np.stack([sigma.real, sigma.imag]).astype(np.float32),
        truth_bands,
        meta,
    )


def simulate(seed, family, size=256, include_background=True):
    if size < 16 or size % 4 or family not in FAMILIES or seed < 0:
        raise ValueError("Invalid simulation shape, family or seed")
    # Exclude effectively empty cuts / isolated subpixel slivers. Do not clip
    # intensities: resample the configuration instead, retaining its RNG seed.
    for attempt in range(64):
        result = _simulate_once(
            seed + attempt * 2**32, family, size, include_background=include_background
        )
        images, grids, sigma, bands, meta = result
        energy = grids[0]
        visible = np.any(
            (bands >= energy[0]) & (bands <= min(float(energy[-1]), 0.0)), axis=0
        )
        second_moment = float(np.mean(images[1].astype(np.float64) ** 2))
        if visible.mean() >= 0.2 and second_moment <= 150:
            meta["rng_seed"] = meta["seed"]
            meta["seed"] = seed
            meta["rejection_attempts"] = attempt
            meta["visible_band_column_fraction"] = float(visible.mean())
            meta["bare_second_moment"] = second_moment
            return result
    raise ValueError(
        f"Could not generate a resolved occupied band for {family}, seed={seed}"
    )


def _simulate_job(job):
    return simulate(*job)


def create_corpus(root, counts, size=256, workers=4):
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Refusing to replace existing corpus {root}")
    if size < 16 or size % 4 or workers < 1 or any(n < 1 for n in counts.values()):
        raise ValueError("Require positive counts/workers and size >=16 divisible by 4")
    root.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "size": size,
        "image_channels": ["clean_measurement", "bareband"],
        "energy_convention": "E_minus_EF_eV",
        "axis_order": "energy,cut_coordinate",
        "train_families": list(TRAIN_FAMILIES),
        "held_out_families": list(HELD_OUT_FAMILIES),
        "splits": {},
    }
    with mp.get_context("spawn").Pool(workers) as pool:
        for number, (split, count) in enumerate(counts.items(), 1):
            folder = root / split
            folder.mkdir()
            families = HELD_OUT_FAMILIES if split == "ood" else TRAIN_FAMILIES
            first_seed = number * 1_000_000
            maps = [
                open_memmap(folder / name, mode="w+", dtype=np.float32, shape=shape)
                for name, shape in [
                    ("images.npy", (count, 2, size, size)),
                    ("grids.npy", (count, 3, size)),
                    ("self_energy.npy", (count, 2, size)),
                    ("bands.npy", (count, 4, size)),
                    ("counts.npy", (count,)),
                ]
            ]
            jobs = (
                (first_seed + i, families[i % len(families)], size)
                for i in range(count)
            )
            with (folder / "parameters.jsonl").open("w") as handle:
                for i, (images, grids, sigma, bands, meta) in enumerate(
                    pool.imap(_simulate_job, jobs, chunksize=4)
                ):
                    for array, value in zip(
                        maps,
                        [images, grids, sigma, bands, meta["counts_per_mean_pixel"]],
                    ):
                        array[i] = value
                    handle.write(json.dumps(meta) + "\n")
                    if (i + 1) % 500 == 0 or i + 1 == count:
                        print(
                            f"{split}: {i+1}/{count}, elapsed={time.perf_counter()-started:.1f}s",
                            flush=True,
                        )
            for array in maps:
                array.flush()
            del maps
            digest = hashlib.sha256(
                (folder / "parameters.jsonl").read_bytes()
            ).hexdigest()
            manifest["splits"][split] = {
                "count": count,
                "first_seed": first_seed,
                "last_seed": first_seed + count - 1,
                "families": list(families),
                "parameters_sha256": digest,
            }
    manifest["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("corpus.py", "band_models.py")
    }
    manifest["generation_seconds"] = time.perf_counter() - started
    # The ready manifest is written only after every array is complete.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Corpus ready: {root}, {manifest['generation_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--ood", type=int, default=1000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    create_corpus(
        args.output,
        {k: getattr(args, k) for k in ("train", "validation", "test", "ood")},
        args.size,
        args.workers,
    )


if __name__ == "__main__":
    main()
