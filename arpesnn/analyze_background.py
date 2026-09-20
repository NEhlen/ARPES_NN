"""Held-out background estimates, preservation controls, and repeated EDC fits."""

import multiprocessing as mp
import argparse
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.special import voigt_profile
from background_data import BackgroundDataset, make_background
from background_model import load_background


def baseline(x, quantile):
    # Conventional energy-dependent lower-envelope estimate; no true signal supplied.
    curve = gaussian_filter1d(
        np.quantile(x, quantile, axis=1), max(1, x.shape[0] * 0.03)
    )
    return np.broadcast_to(curve[:, None], x.shape)


def predict_batch(model, batch, device):
    with torch.inference_mode():
        return model(batch["input"].to(device)).float().cpu().numpy()[:, 0]


def preservation_job(job):
    case, kind, repeat, arrays, centers, widths, true_areas = job
    from benchmark_overlap import fit

    records = []
    # Five EDCs belong to the same noise realization; do not pretend they are independent images.
    for method, array in arrays.items():
        for col in [64, 96, 128, 160, 192]:
            fitted = fit(array[:, col], len(centers))
            order = np.argsort(centers[:, col])
            records.append(
                {
                    "case": case,
                    "background_kind": kind,
                    "repeat": repeat,
                    "method": method,
                    "column": col,
                    "center_mae_meV": float(
                        np.mean(
                            np.abs(np.array(fitted["centers"]) - centers[order, col])
                        )
                        * 1000
                    ),
                    "width_mae_meV": float(
                        np.mean(
                            np.abs(
                                2 * np.array(fitted["gamma"]) - 2 * widths[order, col]
                            )
                        )
                        * 1000
                    ),
                    "area_relative_mae": float(
                        np.mean(
                            np.abs(np.array(fitted["areas"]) - true_areas[order, col])
                            / true_areas[order, col]
                        )
                    ),
                    "fit_success": fitted["success"],
                }
            )
    return records


def fitting_stress(model, device, quantile, output):
    from benchmark_overlap import components, E, SIGMA

    jobs = []
    examples = []
    for case in ["single", "double_0.7", "weak_double", "broad_incoherent"]:
        centers, widths, amps = components(case)
        signal = np.zeros((256, 256), dtype="float32")
        for c, g, a in zip(centers, widths, amps):
            signal += (
                a[None, :]
                * voigt_profile(E[:, None] - c[None, :], SIGMA, g[None, :])
                * 0.06
            )
        signal_scale = float(signal.mean())
        signal /= signal_scale
        true_areas = amps * 0.06 / signal_scale
        for kind in ["none", "hump", "correlated"]:
            background_seed = 800000 + len(jobs)
            bg, _ = make_background(
                signal, np.random.default_rng(background_seed), kind
            )
            for repeat in range(12):
                rng = np.random.default_rng(900000 + len(jobs))
                noisy = (rng.poisson((signal + bg).astype("float64") * 30) / 30).astype(
                    "float32"
                )
                scale = float(noisy.mean())
                with torch.inference_mode():
                    pred = (
                        model(torch.from_numpy(noisy / scale)[None, None].to(device))[
                            0, 0
                        ]
                        .cpu()
                        .numpy()
                        * scale
                    )
                arrays = {
                    "raw": noisy,
                    "network": noisy - pred,
                    "lower_envelope": noisy - baseline(noisy, quantile),
                    "oracle_subtraction": noisy - bg,
                }
                jobs.append((case, kind, repeat, arrays, centers, widths, true_areas))
                if kind == "hump" and repeat == 0:
                    examples.append((case, signal, noisy, bg, pred, arrays["network"]))
    records = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        for i, values in enumerate(pool.map(preservation_job, jobs), 1):
            records.extend(values)
            if i % 24 == 0:
                print(f"Background fitting stress: {i}/{len(jobs)}", flush=True)
    with (output / "fit_records.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    summary = {}
    for case in ["single", "double_0.7", "weak_double", "broad_incoherent"]:
        summary[case] = {}
        for kind in ["none", "hump", "correlated"]:
            summary[case][kind] = {}
            for method in ["raw", "network", "lower_envelope", "oracle_subtraction"]:
                rows = [
                    r
                    for r in records
                    if r["case"] == case
                    and r["background_kind"] == kind
                    and r["method"] == method
                ]
                summary[case][kind][method] = {
                    key: float(np.mean([r[key] for r in rows]))
                    for key in ["center_mae_meV", "width_mae_meV", "area_relative_mae"]
                }
    (output / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    return examples


def analyze(
    corpus, checkpoint, output, device, experimental=None, seed=42, stress=True
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model, saved = load_background(checkpoint, device)
    quantiles = [0.02, 0.05, 0.10, 0.20]
    val = BackgroundDataset(corpus, "validation", seed, saved.get("profile", "v1"))
    scores = np.zeros(len(quantiles))
    for i in range(len(val)):
        item = val[i]
        x = item["input"][0].numpy()
        truth = item["background"][0].numpy()
        for j, q in enumerate(quantiles):
            scores[j] += np.mean((baseline(x, q) - truth) ** 2)
    q = quantiles[int(np.argmin(scores))]
    result = {
        "seed": seed,
        "baseline_quantile": q,
        "validation_baseline_scores": dict(
            zip(map(str, quantiles), (scores / len(val)).tolist())
        ),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "splits": {},
    }
    examples = []
    for split in ["test", "ood"]:
        ds = BackgroundDataset(corpus, split, seed, saved.get("profile", "v1"))
        params = [
            json.loads(s)
            for s in (Path(corpus) / split / "parameters.jsonl")
            .read_text()
            .splitlines()
        ]
        records = []
        chosen = set()
        for batch in DataLoader(ds, batch_size=32):
            predictions = predict_batch(model, batch, device)
            for j, pred in enumerate(predictions):
                x = batch["input"][j, 0].numpy()
                bg = batch["background"][j, 0].numpy()
                signal = batch["signal"][j, 0].numpy()
                index = int(batch["index"][j])
                zero = bool(batch["zero"][j])
                kind = batch["kind"][j]
                estimators = {
                    "none": np.zeros_like(x),
                    "network": pred,
                    "lower_envelope": baseline(x, q),
                }
                for name, estimate in estimators.items():
                    records.append(
                        {
                            "index": index,
                            "method": name,
                            "zero": zero,
                            "kind": kind,
                            "feature": params[index]["retained_signal_feature"],
                            "family": params[index]["family"],
                            "background_mse": float(np.mean((estimate - bg) ** 2)),
                            "corrected_mse": float(
                                np.mean((x - estimate - signal) ** 2)
                            ),
                            "signal_area_removed_fraction": float(
                                (estimate - bg).mean() / signal.mean()
                            ),
                            "absolute_area_error_fraction": float(
                                abs((estimate - bg).mean() / signal.mean())
                            ),
                            "negative_corrected_fraction": float(
                                (x - estimate < 0).mean()
                            ),
                        }
                    )
                key = (
                    "zero"
                    if zero
                    else (
                        "broad"
                        if params[index]["retained_signal_feature"] == "broad_flat"
                        else "nonzero"
                    )
                )
                if key not in chosen and len(chosen) < 3:
                    examples.append((f"{split}: {key}", signal, x, bg, pred, x - pred))
                    chosen.add(key)
        with (output / f"{split}_records.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        metrics = {}
        for subset in ["all", "zero", "broad_flat", "weak_band"]:
            metrics[subset] = {}
            for method in ["none", "network", "lower_envelope"]:
                selected = [
                    r
                    for r in records
                    if r["method"] == method
                    and (
                        subset == "all"
                        or subset == "zero"
                        and r["zero"]
                        or r["feature"] == subset
                    )
                ]
                metrics[subset][method] = {
                    "n": len(selected),
                    **{
                        key: (
                            float(np.mean([r[key] for r in selected]))
                            if selected
                            else None
                        )
                        for key in [
                            "background_mse",
                            "corrected_mse",
                            "signal_area_removed_fraction",
                            "absolute_area_error_fraction",
                            "negative_corrected_fraction",
                        ]
                    },
                }
        result["splits"][split] = metrics
        print(f'{split} background analysis: {json.dumps(metrics["all"])}', flush=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    from background_plots import plot_examples

    plot_examples(examples, output / "heldout_examples.png")
    if stress:
        stress_examples = fitting_stress(model, device, q, output)
        plot_examples(stress_examples, output / "fit_examples.png")
    if experimental:
        from apply_background import apply

        exp = apply(
            model,
            experimental,
            output / "experimental",
            device,
            saved["input_shape"][0],
        )
        result["experimental"] = exp
        (output / "results.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Exploratory additive-background estimator",
        "",
        "The corrected spectrum is input minus estimated background, without clipping. This is not denoising: background shot noise remains. Broad intrinsic components were retained as signal during training.",
        "",
        "## Held-out results",
        "",
        "| Split / subset | Method | Background MSE | Mean signal area removed fraction |",
        "|---|---|---:|---:|",
    ]
    for split, metrics in result["splits"].items():
        for subset in ["all", "zero", "broad_flat", "weak_band"]:
            for method in ["none", "network", "lower_envelope"]:
                row = metrics[subset][method]
                if row["n"]:
                    lines.append(
                        f'| {split} / {subset} | {method} | {row["background_mse"]:.5f} | {row["signal_area_removed_fraction"]:.3f} |'
                    )
    lines += [
        "",
        "Positive removed fraction means oversubtraction of signal; negative means residual background. Zero-background controls directly measure false signal removal. Signed bias can cancel across samples; absolute errors are in results.json.",
        "",
        "The conventional baseline is an energy-dependent momentum quantile, smoothed along energy; its quantile is selected on validation only. The zero estimate is also a baseline.",
        "",
        "The fitting stress test compares raw, corrected and exact-background-subtracted spectra at the same noise level. The oracle still contains shot noise. Independent EDC Voigt fits are not a full 2D or Poisson-likelihood analysis. Twelve noise repeats per condition are preliminary; correlated EDCs must not be counted as independent spectra.",
        "",
        "Synthetic labels define the desired decomposition; they cannot establish that an experimental background is extrinsic. Broad features can be ambiguous. Experimental plots have no decomposition ground truth and do not validate the subtraction.",
        "",
        "See heldout_examples.png, fit_summary.json / fit_examples.png (when enabled), and experimental/comparison.png (when supplied).",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experimental", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--skip-stress", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    analyze(
        args.corpus,
        args.checkpoint,
        args.output,
        torch.device(args.device),
        args.experimental,
        stress=not args.skip_stress,
    )


if __name__ == "__main__":
    main()
