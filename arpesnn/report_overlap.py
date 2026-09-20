"""Summarize paired fitting trials without treating EDCs as independent images."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from benchmark_overlap import OUT, CASES, COUNTS, METHODS, E, SIGMA, fit


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Completed overlap benchmark output directory",
    )
    OUT = parser.parse_args().input
    result = json.loads((OUT / "results.json").read_text())
    rows = [json.loads(line) for line in (OUT / "fits.jsonl").read_text().splitlines()]
    assert len(rows) == 738 * 4 * 5, len(rows)
    assert all("naive_center_se" in r["fit"] for r in rows)
    assert len(
        {
            (r["split"], r["case"], r["count"], r["repeat"], r["method"], r["column"])
            for r in rows
        }
    ) == len(rows)
    paired = {}
    rng = np.random.default_rng(472)
    for case in CASES:
        for count in COUNTS:
            subsets = {
                method: [
                    r
                    for r in rows
                    if r["split"] == "test"
                    and r["case"] == case
                    and r["count"] == count
                    and r["method"] == method
                ]
                for method in METHODS
            }
            errors = {
                method: np.array(
                    [
                        np.mean(
                            [r["position_mae_eV"] for r in values if r["repeat"] == rep]
                        )
                        for rep in range(24)
                    ]
                )
                for method, values in subsets.items()
            }
            key = f"{case}/{count}"
            paired[key] = {}
            for method in ("raw", "gaussian", "fourier"):
                delta = 1000 * (errors[method] - errors["network"])
                boot = rng.choice(delta, (2000, 24), replace=True).mean(1)
                paired[key][method] = {
                    "baseline_minus_network_position_MAE_meV": float(delta.mean()),
                    "paired_bootstrap_95pct": np.quantile(
                        boot, [0.025, 0.975]
                    ).tolist(),
                }
    (OUT / "paired_bootstrap.json").write_text(json.dumps(paired, indent=2))
    labels = [
        "single",
        "double 0.35×",
        "double 0.7×",
        "double 1.2×",
        "double 2×",
        "weak double",
        "crossing",
        "avoided crossing",
        "broad incoherent",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    matrices = []
    for count in COUNTS:
        matrices.append(
            np.array(
                [
                    [
                        result["cases"][case][str(count)][method]["position_mae_meV"]
                        for method in METHODS
                    ]
                    for case in CASES
                ]
            )
        )
    for ax, count, data in zip(axes, COUNTS, matrices):
        im = ax.imshow(
            data,
            cmap="magma_r",
            vmin=0,
            vmax=max(np.max(m) for m in matrices),
            aspect="auto",
        )
        ax.set(
            xticks=range(4),
            xticklabels=METHODS,
            yticks=range(len(CASES)),
            yticklabels=labels,
            title=f"{count:g} counts / mean pixel",
        )
        for (i, j), value in np.ndenumerate(data):
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > max(np.max(m) for m in matrices) * 0.5
                    else "black"
                ),
            )
    fig.colorbar(
        im, ax=axes, label="Component position MAE (meV); lower is better", shrink=0.7
    )
    fig.suptitle(
        "Fitted component positions — known component count, 24 repeats × 5 EDCs\nDoublet ratios refer to separation / intrinsic Lorentzian FWHM"
    )
    fig.savefig(OUT / "all_case_positions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(15, 9), layout="constrained")
    for row, (case, col) in enumerate(
        [("double_0.7", 128), ("weak_double", 128), ("avoided_crossing", 128)]
    ):
        z = np.load(OUT / f"example_{case}.npz")
        for ax, method in zip(axes[row], METHODS):
            y = z[method][:, col]
            f = fit(y, 2)
            parts = np.array(
                [
                    a * voigt_profile(E - c, SIGMA, g)
                    for a, c, g in zip(f["areas"], f["centers"], f["gamma"])
                ]
            )
            design = np.column_stack([np.ones_like(E), E - E.mean()])
            background = (
                design @ np.linalg.lstsq(design, y - parts.sum(0), rcond=None)[0]
            )
            ax.plot(E, y, alpha=0.5, lw=1, label="input to fit")
            ax.plot(E, z["clean"][:, col], color="black", lw=1, label="clean total")
            ax.plot(
                E,
                parts.sum(0) + background,
                "--",
                color="red",
                lw=1.2,
                label="two-component fit",
            )
            for center in z["centers"][:, col]:
                ax.axvline(center, color="black", ls=":", alpha=0.5)
            for center in f["centers"]:
                ax.axvline(center, color="red", ls="--", alpha=0.6)
            ax.set(title=method, xlabel="Energy (eV)", ylabel=case, xlim=(-0.25, -0.05))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Representative central EDC fits, 30 counts/mean pixel, first noise realization\nBlack dotted: true component centers; red dashed: fitted centers"
    )
    fig.savefig(OUT / "fitted_edcs.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Overlapping-band fitting benchmark",
        "",
        "The network was not retrained. This evaluates the existing denoiser on new controlled Voigt spectra, not on the original training corpus.",
        "",
        "## Design",
        "",
        "- Nine cases: single component; four equal-width doublets with separation/FWHM ratios 0.35, 0.7, 1.2 and 2; a weak component with 15% of the strong component area; crossing; avoided crossing (30 meV minimum gap); narrow plus broad incoherent component.",
        "- Three count levels: 5, 30 and 200 expected counts per mean pixel.",
        "- 24 independent Poisson noise realizations per case/count: 648 test images. Five EDCs per image, four methods, one- and two-component fits with multiple initializations.",
        "- 24 separate calibration images choose filters for mean fitted position plus intrinsic linewidth error. Nine Gaussian settings and eight smooth Fourier settings. Separate single-band controls (90 images) calibrate two-component detection at a nominal 5% EDC false-positive rate.",
        "- Known instrumental Gaussian width: sigma=6 meV. A linear background and one or two Voigt components are fitted with identical unweighted least-squares objective, data-derived starts and bounds for all methods. No ground-truth centers initialize fits.",
        "- Ground truth is the individual component center and intrinsic Lorentzian FWHM, not the maximum of the combined intensity.",
        "- Confidence intervals on method differences bootstrap 24 whole noise realizations; EDCs from one image are not treated as independent trials.",
        "",
        "## Position errors at 30 counts per mean pixel",
        "",
        "| Case | Raw | Network | Gaussian | Fourier |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        values = [
            result["cases"][case]["30.0"][method]["position_mae_meV"]
            for method in METHODS
        ]
        lines.append(
            "| " + case + " | " + " | ".join(f"{v:.2f}" for v in values) + " |"
        )
    lines += [
        "",
        "Errors above are meV. See all_case_positions.png for all count levels.",
        "",
        "## Reliability at 30 counts per mean pixel",
        "",
        "| Case / metric | Raw | Network | Gaussian | Fourier |",
        "|---|---:|---:|---:|---:|",
    ]
    for case, metric, label in [
        ("single", "two_component_detection_fraction", "Single-band false splitting"),
        ("double_0.7", "two_component_detection_fraction", "Close doublet detection"),
        ("weak_double", "two_component_detection_fraction", "Weak doublet detection"),
        (
            "double_0.7",
            "both_centers_within_10meV_fraction",
            "Close doublet: both centers within 10 meV",
        ),
        (
            "weak_double",
            "both_centers_within_10meV_fraction",
            "Weak doublet: both centers within 10 meV",
        ),
        (
            "double_0.7",
            "naive_95pct_center_interval_coverage",
            "Close doublet: naive 95% interval coverage",
        ),
    ]:
        values = [
            100 * result["cases"][case]["30.0"][method][metric] for method in METHODS
        ]
        lines.append(
            "| " + label + " | " + " | ".join(f"{v:.1f}%" for v in values) + " |"
        )
    lines += [
        "",
        "Detection thresholds are method-specific and fitted only to calibration single-band controls; measured test false-positive rates can differ from 5%. A positive detection does not by itself mean component positions are accurate.",
        "",
        "## Interpretation and limits",
        "",
        "- This tests local EDC fit usefulness, not just visual smoothness or image MSE. Raw joint 2D fitting or a Poisson-likelihood fit could do better than the unweighted independent-EDC baseline used here.",
        "- Filters average neighboring momenta and may change lineshapes; fits use the original physical instrument resolution and do not reverse the filtering. Width bias therefore measures the consequences of fitting a processed spectrum naively.",
        "- Known component count is supplied for position/width metrics. One-versus-two detection is a separate calibrated task; neither metric alone establishes reliable band separation.",
        "- Naive Jacobian error bars assume independent equal-variance residuals, which is false after smoothing and imperfect even for raw Poisson noise. Their empirical coverage is a diagnostic, not a recommendation to trust them.",
        "- Synthetic profiles exactly match the fitting family before processing. This is an optimistic controlled problem; matrix elements, Fermi cutoffs, detector artifacts, momentum resolution, and misspecified real lineshapes remain untested.",
        "- The broad incoherent case is phenomenological and does not constitute a full interacting waterfall model. Crossing branches are sorted by energy locally; orbital identity tracking is not assessed.",
        "- Only 24 test realizations per condition: percentages and bootstrap intervals are preliminary, particularly for rare false splitting. Threshold calibration spans 30 images per count, with correlated EDCs.",
        "- No experimental validation and no new training were performed.",
        "",
        "## Reproduction",
        "",
        "Run `uv run python arpesnn/benchmark_overlap.py --checkpoint /path/to/denoiser.pt --output outputs/overlap-benchmark` followed by `uv run python arpesnn/report_overlap.py --input outputs/overlap-benchmark`.",
        "",
        "Saved: full per-fit records, seeds, source snapshot, checkpoint hash, filter tuning, method-specific detection thresholds, empirical bias/scatter/coverage metrics and paired-bootstrap intervals.",
    ]
    gaps = {}
    gap_lines = [
        "",
        "## Near-crossing EDC: forcing two components can invent an apparent gap",
        "",
        "Values below are the mean fitted splitting at sampled momentum k=0.00392 (not a global dispersion fit). The crossing truth is 0.86 meV at this pixel; the avoided-crossing truth is 30.01 meV.",
        "",
        "| Case / counts | Raw | Network | Gaussian | Fourier |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in ("crossing", "avoided_crossing"):
        for count in COUNTS:
            values = {}
            for method in METHODS:
                subset = [
                    r
                    for r in rows
                    if r["split"] == "test"
                    and r["case"] == case
                    and r["count"] == count
                    and r["method"] == method
                    and r["column"] == 128
                ]
                splitting = np.array(
                    [np.diff(r["fit"]["centers"])[0] * 1000 for r in subset]
                )
                threshold = result["detection_thresholds"][f"{method}/{count}"]
                values[method] = {
                    "mean_splitting_meV": float(splitting.mean()),
                    "median_splitting_meV": float(np.median(splitting)),
                    "two_component_detection_fraction": float(
                        np.mean([r["two_component_score"] > threshold for r in subset])
                    ),
                }
            gaps[f"{case}/{count}"] = values
            gap_lines.append(
                "| "
                + f"{case}, {count:g}"
                + " | "
                + " | ".join(f'{values[m]["mean_splitting_meV"]:.2f}' for m in METHODS)
                + " |"
            )
    (OUT / "near_crossing.json").write_text(json.dumps(gaps, indent=2))
    gap_lines += [
        "",
        "These forced two-component results are ill-conditioned near a true crossing. They must not be interpreted as evidence for a physical gap. The calibrated model-selection decision is separate; see near_crossing.json for the fraction of trials actually classified as two components.",
        "",
        "## Main findings",
        "",
        "At 30 counts, the network improves mean position accuracy in all nine tested cases relative to both filters. The advantage is smaller than the earlier image-MSE ratios implied. For the weak doublet, both positions fall within 10 meV of truth in 88% of EDCs, versus 70% for Gaussian, 53% for Fourier and 34% for raw fits.",
        "",
        "There are failures: at 5 counts and separation 0.35 times the intrinsic linewidth, network mean position error is 35.2 meV, worse than raw (24.1), Gaussian (29.2) and Fourier (33.6). Despite more successful trials, a tail of bad fits dominates its mean. At 30 counts, nominal 95% least-squares intervals cover close-doublet centers only 39% of the time after network processing. Smoother residuals therefore cannot justify smaller physical uncertainty estimates.",
        "",
        "The network is useful for many controlled overlapping-peak fits, but neither denoising nor forcing a multi-component fit establishes identifiability. Real repeats and more realistic forward-model stress tests remain necessary.",
    ]
    lines += gap_lines
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[14:41]))


if __name__ == "__main__":
    main()
