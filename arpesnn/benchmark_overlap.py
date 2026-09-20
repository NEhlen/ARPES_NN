"""Repeated-noise downstream fitting benchmark; no retraining or test tuning.

Voigt EDC components, known instrumental Gaussian width, linear background.
Processed-curve fit scores are empirically calibrated, not interpreted as BIC.
"""

import argparse
import json
import hashlib
import shutil
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import voigt_profile
from scipy.optimize import least_squares
import torch
from apply_model import load_model
from benchmark_denoising import filtered

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "arpesnn/models/training_v2_20260919"
OUT = RUN / "overlap_benchmark"
E = np.linspace(-0.45, 0.15, 256)
K = np.linspace(-1, 1, 256)
SIGMA = 0.006
COLS = [64, 96, 128, 160, 192]
COUNTS = [5.0, 30.0, 200.0]
CASES = [
    "single",
    "double_0.35",
    "double_0.7",
    "double_1.2",
    "double_2.0",
    "weak_double",
    "crossing",
    "avoided_crossing",
    "broad_incoherent",
]
METHODS = ["raw", "network", "gaussian", "fourier"]
FILTERS = [
    ("gaussian", a, b, 0)
    for a, b in (
        (0.4, 0.4),
        (0.7, 0.4),
        (1.0, 0.7),
        (1.5, 1.0),
        (2.0, 1.5),
        (3.0, 2.0),
        (4.0, 3.0),
        (6.0, 4.0),
        (8.0, 6.0),
    )
] + [
    ("fourier", a, b, 4)
    for a, b in (
        (0.04, 0.06),
        (0.06, 0.1),
        (0.08, 0.15),
        (0.12, 0.2),
        (0.2, 0.3),
        (0.3, 0.3),
        (0.3, 0.5),
        (0.5, 0.5),
    )
]


def components(case, variant=0):
    # Independent calibration geometries differ slightly from evaluation geometries.
    center = -0.15 + 0.04 * K + 0.025 * K * K + variant * 0.008
    gamma = np.full_like(K, 0.022 + variant * 0.002)
    if case == "single":
        centers, widths, amps = center[None], gamma[None], np.ones((1, len(K)))
    elif case.startswith("double_") or case == "weak_double":
        ratio = float(case.split("_")[1]) if case.startswith("double_") else 0.8
        separation = ratio * 2 * gamma
        centers = np.stack([center - separation / 2, center + separation / 2])
        widths = np.stack([gamma, gamma])
        amps = np.stack(
            [np.ones_like(K), np.full_like(K, 0.15 if case == "weak_double" else 0.8)]
        )
    elif case in ("crossing", "avoided_crossing"):
        gap = 0.015 if case == "avoided_crossing" else 0.0
        d = np.sqrt((0.11 * K) ** 2 + gap**2)
        centers = np.stack([center - d, center + d])
        widths = np.stack([gamma, gamma])
        amps = np.stack([np.ones_like(K), np.full_like(K, 0.65)])
    else:
        # Phenomenological coherent + broad incoherent overlapping component.
        # This is not claimed to be a microscopic waterfall spectral function.
        centers = np.stack([center, center - 0.05 - 0.05 * np.tanh(4 * K)])
        widths = np.stack([gamma, 0.045 + 0.04 / (1 + np.exp(-6 * K))])
        amps = np.stack([np.ones_like(K), np.full_like(K, 0.6)])
    return centers, widths, amps


def synth(case, variant=0):
    centers, widths, amps = components(case, variant)
    signal = np.full((len(E), len(K)), 0.35) + 0.15 * (E[:, None] - E.mean()) / (
        E.ptp() if hasattr(E, "ptp") else np.ptp(E)
    )
    for c, g, a in zip(centers, widths, amps):
        signal += (
            a[None, :]
            * voigt_profile(E[:, None] - c[None, :], SIGMA, g[None, :])
            * 0.06
        )
    scale = signal.mean()
    return (signal / scale).astype("float32"), centers, widths, amps


def curve(parameters, number):
    out = parameters[-2] + parameters[-1] * (E - E.mean())
    for j in range(number):
        area, center, gamma = parameters[3 * j : 3 * j + 3]
        out = out + area * voigt_profile(E - center, SIGMA, gamma)
    return out


def fit(y, number):
    # Same unweighted objective, bounds, and data-derived starts for every method.
    # Empirical repeated-noise errors, not least-squares covariance, are reported.
    y = np.asarray(y, dtype=float)
    base = max(float(np.percentile(y, 15)), 0.001)
    weights = np.maximum(y - base, 0)
    centroid = float(np.sum(E * weights) / max(weights.sum(), 1e-12))
    peak = float(E[np.argmax(y)])
    starts = (
        [(peak, 0.03)] if number == 1 else [(centroid, s) for s in (0.015, 0.05, 0.10)]
    )
    best = None
    lower = [0.0, E[2], 0.003] * number + [0.0, -2.0]
    upper = [5.0, E[-3], 0.16] * number + [5.0, 2.0]
    for mid, spread in starts:
        centers = [mid] if number == 1 else [mid - spread / 2, mid + spread / 2]
        p = []
        for c in centers:
            p.extend(
                [
                    max(np.trapezoid(weights, E) / number, 0.002),
                    np.clip(c, E[3], E[-4]),
                    0.025,
                ]
            )
        p += [min(base, 4.9), 0.0]
        result = least_squares(
            lambda v: curve(v, number) - y,
            np.clip(p, np.array(lower) + 1e-8, np.array(upper) - 1e-8),
            bounds=(lower, upper),
            max_nfev=150,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )
        sse = float(np.sum(result.fun**2))
        if best is None or sse < best["sse"]:
            values = result.x[: 3 * number].reshape(number, 3)
            order = np.argsort(values[:, 1])
            covariance = (
                np.linalg.pinv(result.jac.T @ result.jac)
                * sse
                / (len(E) - len(result.x))
            )
            center_se = np.sqrt(np.maximum(np.diag(covariance)[1 : 3 * number : 3], 0))[
                order
            ]
            values = values[order]
            best = {
                "sse": sse,
                "success": bool(result.success),
                "naive_center_se": center_se.tolist(),
                "centers": values[:, 1].tolist(),
                "gamma": values[:, 2].tolist(),
                "areas": values[:, 0].tolist(),
            }
    return best


def analyze_job(job):
    meta, arrays, truth_centers, truth_widths = job
    output = []
    n = len(truth_centers)
    for method, y in arrays.items():
        for col in COLS:
            one = fit(y[:, col], 1)
            two = fit(y[:, col], 2)
            chosen = one if n == 1 else two
            centers = np.sort(truth_centers[:, col])
            order = np.argsort(truth_centers[:, col])
            widths = truth_widths[order, col]
            score = float(
                len(E) * np.log(max(one["sse"], 1e-20) / max(two["sse"], 1e-20))
            )
            output.append(
                {
                    **meta,
                    "method": method,
                    "column": col,
                    "true_centers": centers.tolist(),
                    "true_gamma": widths.tolist(),
                    "fit": chosen,
                    "one": one,
                    "two": two,
                    "two_component_score": score,
                    "position_mae_eV": float(
                        np.mean(np.abs(np.asarray(chosen["centers"]) - centers))
                    ),
                    "linewidth_mae_eV": float(
                        np.mean(np.abs(2 * np.asarray(chosen["gamma"]) - 2 * widths))
                    ),
                    "splitting_error_eV": (
                        float(abs(np.diff(chosen["centers"])[0] - np.diff(centers)[0]))
                        if n == 2
                        else None
                    ),
                }
            )
    return output


def tune_job(job):
    case, count, seed = job
    clean, centers, widths, _ = synth(case, variant=1)
    rng = np.random.default_rng(seed)
    x = (rng.poisson(clean * count) / count).astype("float32")
    scores = []
    for candidate in FILTERS:
        y = filtered(x, candidate)
        errors = []
        for col in (96, 160):
            f = fit(y[:, col], len(centers))
            order = np.argsort(centers[:, col])
            errors.append(
                np.mean(np.abs(np.array(f["centers"]) - centers[order, col]))
                + np.mean(np.abs(2 * np.array(f["gamma"]) - 2 * widths[order, col]))
            )
        scores.append(float(np.mean(errors)))
    return count, scores


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, OUT / "benchmark_source.py")
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(
                    (args.checkpoint).read_bytes()
                ).hexdigest(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "energy_eV": E.tolist(),
                "momentum_coordinate": K.tolist(),
                "edc_columns": COLS,
                "seed_policy": "calibration >= 200000; test >= 300000; exact seed recorded per image",
            },
            indent=2,
        )
    )
    torch.set_num_threads(2)
    settings = {}
    tuning = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs = [
            (case, count, 10000 + i * 100 + j * 10 + r)
            for i, case in enumerate(
                ("single", "double_0.7", "weak_double", "avoided_crossing")
            )
            for j, count in enumerate(COUNTS)
            for r in range(2)
        ]
        for count, scores in pool.map(tune_job, jobs):
            tuning.append((count, scores))
            print(f"Tuning {len(tuning)}/{len(jobs)}", flush=True)
    for count in COUNTS:
        mean = np.mean([v for c, v in tuning if c == count], axis=0)
        settings[str(count)] = {
            "gaussian": FILTERS[int(np.argmin(mean[:9]))],
            "fourier": FILTERS[9 + int(np.argmin(mean[9:]))],
        }
    (OUT / "settings.json").write_text(
        json.dumps(
            {
                "filters": settings,
                "candidate_filters": FILTERS,
                "validation_scores": tuning,
                "objective": "mean center MAE + Lorentzian FWHM MAE, known component number; distinct calibration geometries and seeds",
            },
            indent=2,
        )
    )
    model, _ = load_model(args.checkpoint, torch.device("cpu"))
    records = []
    stream = (OUT / "fits.jsonl").open("w")

    def collect(future):
        values = future.result()
        records.extend(values)
        for record in values:
            stream.write(json.dumps(record) + "\n")
        stream.flush()

    # Separate single-peak calibration trials provide empirical 5% false-positive thresholds.
    tasks = [("calibration", "single", c, r) for c in COUNTS for r in range(30)]
    tasks += [("test", case, c, r) for case in CASES for c in COUNTS for r in range(24)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = []
        for ix, (split, case, count, repeat) in enumerate(tasks):
            clean, centers, widths, _ = synth(
                case, variant=1 if split == "calibration" else 0
            )
            seed = (200000 if split == "calibration" else 300000) + ix
            rng = np.random.default_rng(seed)
            x = (rng.poisson(clean * count) / count).astype("float32")
            scale = float(x.mean())
            with torch.inference_mode():
                pred = (
                    model(torch.from_numpy(x / scale)[None, None])[0, 0].numpy() * scale
                )
            arrays = {
                "raw": x,
                "network": pred,
                **{
                    m: filtered(x, settings[str(count)][m])
                    for m in ("gaussian", "fourier")
                },
            }
            meta = {
                "split": split,
                "case": case,
                "count": count,
                "repeat": repeat,
                "seed": seed,
            }
            pending.append(pool.submit(analyze_job, (meta, arrays, centers, widths)))
            if split == "test" and repeat == 0 and count == 30:
                np.savez_compressed(
                    OUT / f"example_{case}.npz",
                    clean=clean,
                    centers=centers,
                    widths=widths,
                    **arrays,
                )
            if len(pending) >= 12:
                collect(pending.pop(0))
            if (ix + 1) % 24 == 0:
                print(
                    f"Images submitted {ix+1}/{len(tasks)}; fitted EDCs {len(records)}",
                    flush=True,
                )
        for future in pending:
            collect(future)
    stream.close()
    temporary = OUT / "fits.complete.jsonl"
    with temporary.open("w") as final_stream:
        for record in records:
            final_stream.write(json.dumps(record) + "\n")
    temporary.replace(OUT / "fits.jsonl")
    summarize(records)


def summarize(records):
    summary = {}
    thresholds = {}
    for method in METHODS:
        for count in COUNTS:
            vals = [
                r["two_component_score"]
                for r in records
                if r["split"] == "calibration"
                and r["method"] == method
                and r["count"] == count
            ]
            thresholds[(method, count)] = float(np.quantile(vals, 0.95))
    for case in CASES:
        summary[case] = {}
        for count in COUNTS:
            summary[case][str(count)] = {}
            for method in METHODS:
                rows = [
                    r
                    for r in records
                    if r["split"] == "test"
                    and r["case"] == case
                    and r["count"] == count
                    and r["method"] == method
                ]
                position = np.array([r["position_mae_eV"] for r in rows])
                width = np.array([r["linewidth_mae_eV"] for r in rows])
                called = np.array(
                    [
                        r["two_component_score"] > thresholds[(method, count)]
                        for r in rows
                    ]
                )
                # Confidence coverage is empirical: actual errors and repeat scatter are saved;
                # do not call correlated-pixel fit covariances valid uncertainties.
                bias = []
                scatter = []
                for col in COLS:
                    group = [r for r in rows if r["column"] == col]
                    errors = np.array(
                        [
                            np.array(r["fit"]["centers"]) - r["true_centers"]
                            for r in group
                        ]
                    )
                    bias.extend(np.abs(errors.mean(0)))
                    scatter.extend(errors.std(0, ddof=1))
                summary[case][str(count)][method] = {
                    "position_mae_meV": float(position.mean() * 1000),
                    "linewidth_mae_meV": float(width.mean() * 1000),
                    "absolute_bias_meV": float(np.mean(bias) * 1000),
                    "repeat_std_meV": float(np.mean(scatter) * 1000),
                    "two_component_detection_fraction": float(called.mean()),
                    "fit_nonconvergence_fraction": float(
                        np.mean([not r["fit"]["success"] for r in rows])
                    ),
                    "both_centers_within_10meV_fraction": float(
                        np.mean(
                            [
                                np.max(
                                    np.abs(
                                        np.array(r["fit"]["centers"])
                                        - r["true_centers"]
                                    )
                                )
                                < 0.01
                                for r in rows
                            ]
                        )
                    ),
                    "splitting_mae_meV": (
                        float(np.mean([r["splitting_error_eV"] for r in rows]) * 1000)
                        if case != "single"
                        else None
                    ),
                    "naive_95pct_center_interval_coverage": float(
                        np.mean(
                            [
                                np.mean(
                                    np.abs(
                                        np.array(r["fit"]["centers"])
                                        - r["true_centers"]
                                    )
                                    <= 1.96 * np.array(r["fit"]["naive_center_se"])
                                )
                                for r in rows
                            ]
                        )
                    ),
                    "n_noise_repeats": 24,
                    "n_edcs": len(rows),
                }
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "cases": summary,
                "detection_thresholds": {
                    f"{m}/{c}": v for (m, c), v in thresholds.items()
                },
                "notes": [
                    "Known-number fits and empirically calibrated number detection are separate tests.",
                    "Noise realizations are independent; EDCs from one image are correlated.",
                    "Naive intervals use iid least-squares Jacobian covariance; empirical coverage tests that assumption and does not endorse it.",
                    "Only Poisson noise and exactly specified Voigt forward models tested.",
                    "Broad incoherent case is phenomenological, not a microscopic waterfall.",
                    "Instrumental Gaussian sigma fixed at true 6 meV for all methods.",
                    "Widths are intrinsic Lorentzian FWHM, not total Voigt FWHM.",
                ],
            },
            indent=2,
        )
    )
    make_plots(summary)
    print("Benchmark complete", flush=True)


def make_plots(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ["double_0.35", "double_0.7", "double_1.2", "double_2.0"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, count in zip(axes, COUNTS):
        for method in METHODS:
            ax.plot(
                [0.35, 0.7, 1.2, 2.0],
                [summary[c][str(count)][method]["position_mae_meV"] for c in cases],
                "-o",
                label=method,
            )
        ax.set(
            title=f"{count:g} counts / mean pixel",
            xlabel="Component separation / Lorentzian FWHM",
            ylabel="Fitted component position MAE (meV)",
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "Known two-component fits, 24 noise realizations × 5 EDCs per case; lower is better"
    )
    fig.savefig(OUT / "position_error.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(17, 12), layout="constrained")
    for row, case in enumerate(
        ("double_0.7", "weak_double", "avoided_crossing", "broad_incoherent")
    ):
        z = np.load(OUT / f"example_{case}.npz")
        for ax, method in zip(axes[row], ["clean"] + METHODS):
            ax.plot(E, z[method][:, 128], label=method)
            for center in z["centers"][:, 128]:
                ax.axvline(center, color="k", ls=":", alpha=0.5)
            ax.set(title=method, xlabel="Energy (eV)", ylabel=case, xlim=(-0.3, 0))
        axes[row, 0].set_title("Clean reference")
    fig.suptitle(
        "Central EDCs at 30 counts / mean pixel; dotted lines are actual component centers"
    )
    fig.savefig(OUT / "edc_examples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
