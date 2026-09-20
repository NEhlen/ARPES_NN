"""Toy band Hamiltonians for controlled generalization tests, not material fits.

Energies are eV, k is inverse angstrom, and lattice/bond lengths are angstrom.
All arrays use bands x momentum. The 2x2 eigenvalues are evaluated analytically;
the Bernal bilayer uses batched Hermitian diagonalization.
"""

import numpy as np

FAMILIES = (
    "parabolic",
    "multiband_parabolic",
    "dirac",
    "bilayer",
    "mexican_hat",
    "spin_split_valence",
    "square_lattice",
    "honeycomb",
    "avoided_crossing",
    "saddle",
    "triangular_lattice",
    "rashba",
)
HELD_OUT_FAMILIES = ("mexican_hat", "saddle")
TRAIN_FAMILIES = tuple(f for f in FAMILIES if f not in HELD_OUT_FAMILIES)


def sample_parameters(rng, family):
    if family not in FAMILIES:
        raise ValueError(f"Unknown band family {family}")
    return {
        "family": family,
        "shift": 0.0,
        "curvature_x": float(rng.uniform(0.25, 4.0)),
        "curvature_y": float(rng.uniform(0.25, 4.0)),
        "velocity": float(rng.uniform(0.6, 5.0)),
        "gap": float(rng.uniform(0, 0.3)),
        "split": float(rng.uniform(0.08, 0.65)),
        "coupling": float(rng.uniform(0.02, 0.25)),
        "hopping_x": float(rng.uniform(0.15, 1.8)),
        "hopping_y": float(rng.uniform(0.15, 1.8)),
        "hopping_next": float(rng.uniform(-0.3, 0.3)),
        "lattice_a": float(rng.uniform(1.3, 3.2)),
        "radius": float(rng.uniform(0.12, 0.65)),
        "quartic": float(rng.uniform(0.5, 8.0)),
        "band_count": int(rng.integers(2, 5)),
        "mass_sign": float(rng.choice([-1.0, 1.0])),
    }


def band_energies(kx, ky, p):
    family = p["family"]
    cx, cy = p["curvature_x"], p["curvature_y"]
    r2 = kx**2 + ky**2
    mass = cx * kx**2 + cy * ky**2
    a = p["lattice_a"]
    tx, ty = p["hopping_x"], p["hopping_y"]
    if family == "parabolic":
        bands = (p["mass_sign"] * mass)[None, :]
    elif family == "multiband_parabolic":
        bands = np.array(
            [
                p["mass_sign"] * mass * (1 + 0.25 * i) + i * p["split"]
                for i in range(p["band_count"])
            ]
        )
    elif family == "dirac":
        v = np.sqrt(p["velocity"] ** 2 * r2 + (p["gap"] / 2) ** 2)
        bands = np.array([-v, v])
    elif family == "bilayer":
        # Minimal four-orbital Bernal model, omitting trigonal warping.
        h = np.zeros((len(kx), 4, 4), dtype=np.complex128)
        z = p["velocity"] * (kx + 1j * ky)
        h[:, 0, 1] = h[:, 2, 3] = z
        h[:, 1, 0] = h[:, 3, 2] = z.conj()
        h[:, 1, 2] = h[:, 2, 1] = p["split"]
        h[:, 0, 0] = h[:, 1, 1] = p["gap"] / 2
        h[:, 2, 2] = h[:, 3, 3] = -p["gap"] / 2
        bands = np.linalg.eigvalsh(h).T
    elif family == "mexican_hat":
        bands = (p["quartic"] * (r2 - p["radius"] ** 2) ** 2)[None, :]
    elif family == "spin_split_valence":
        bands = np.array([-mass, -mass - p["split"]])
    elif family == "square_lattice":
        bands = (
            -2 * tx * np.cos(a * kx)
            - 2 * ty * np.cos(a * ky)
            - 4 * p["hopping_next"] * np.cos(a * kx) * np.cos(a * ky)
        )[None, :]
    elif family == "honeycomb":
        # a is nearest-neighbor bond length in this family.
        f = np.exp(1j * a * ky) + 2 * np.exp(-0.5j * a * ky) * np.cos(
            np.sqrt(3) * a * kx / 2
        )
        v = np.sqrt(tx**2 * abs(f) ** 2 + (p["gap"] / 2) ** 2)
        bands = np.array([-v, v])
    elif family == "avoided_crossing":
        e1 = p["velocity"] * kx + cy * ky**2
        e2 = -0.6 * p["velocity"] * kx + cx * ky**2 + p["split"]
        center = (e1 + e2) / 2
        split = np.sqrt(((e1 - e2) / 2) ** 2 + p["coupling"] ** 2)
        bands = np.array([center - split, center + split])
    elif family == "saddle":
        bands = (cx * kx**2 - cy * ky**2)[None, :]
    elif family == "triangular_lattice":
        bands = (
            -2
            * tx
            * (
                np.cos(a * kx)
                + 2 * np.cos(a * kx / 2) * np.cos(np.sqrt(3) * a * ky / 2)
            )
        )[None, :]
    elif family == "rashba":
        split = np.sqrt(p["velocity"] ** 2 * r2 + (p["gap"] / 2) ** 2)
        bands = np.array([mass - split, mass + split])
    else:
        raise ValueError(f"Unknown band family {family}")
    return bands + p["shift"]
