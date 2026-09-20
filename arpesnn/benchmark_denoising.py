"""Validation-tuned conventional filters versus the fixed trained denoiser."""

import argparse
from functools import partial
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import dctn, idctn
import torch
from apply_model import load_model
from train_tasks import CorpusDataset

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "arpesnn/models/training_v2_20260919"
OUT = RUN / "filter_benchmark"
SIGMAS = [0, 0.25, 0.4, 0.6, 0.8, 1.2, 2, 3]
CUTOFFS = [0.04, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0]
CANDIDATES = [("gaussian", a, b, 0) for a in SIGMAS for b in SIGMAS] + [
    ("fourier", a, b, n) for a in CUTOFFS for b in CUTOFFS for n in (2, 4)
]


def transfer(shape, candidate):
    _, a, b, n = candidate
    # DCT-II is the Fourier transform of an even extension: avoids wraparound.
    fy = np.arange(shape[0], dtype=np.float32)[:, None] / (2 * shape[0])
    fx = np.arange(shape[1], dtype=np.float32)[None, :] / (2 * shape[1])
    return 1 / (1 + ((fy / a) ** 2 + (fx / b) ** 2) ** n)


def filtered(x, candidate, spectrum=None, response=None):
    kind, a, b, _ = candidate
    if kind == "gaussian":
        result = gaussian_filter(x, (a, b), mode="reflect")
    else:
        if spectrum is None:
            spectrum = dctn(x, norm="ortho")
        result = idctn(
            spectrum * (transfer(x.shape, candidate) if response is None else response),
            norm="ortho",
        )
    return np.maximum(result, 0)


def tune_chunk(indices, corpus, seed):
    ds = CorpusDataset(corpus, "validation", seed)
    scores = np.zeros((len(indices), len(CANDIDATES)), dtype=np.float64)
    responses = {
        c: transfer((ds.manifest["size"], ds.manifest["size"]), c)
        for c in CANDIDATES
        if c[0] == "fourier"
    }
    for row, i in enumerate(indices):
        x, target, _, _ = ds[int(i)]
        x, target = x[0].numpy(), target[0].numpy()
        spectrum = dctn(x, norm="ortho")
        for col, c in enumerate(CANDIDATES):
            y = filtered(x, c, spectrum, responses.get(c))
            scores[row, col] = np.mean((y - target) ** 2, dtype=np.float64)
    return indices, scores


def summary(errors):
    return {name: float(np.mean(values)) for name, values in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    CONFIG = {"seed": args.seed}
    CORPUS = args.corpus
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    nval = len(CorpusDataset(CORPUS, "validation", CONFIG["seed"]))
    scores = np.empty((nval, len(CANDIDATES)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for indices, values in pool.map(
            partial(tune_chunk, corpus=CORPUS, seed=CONFIG["seed"]),
            np.array_split(np.arange(nval), min(nval, 40)),
        ):
            scores[indices] = values
            print(f"Validation through {indices[-1]+1}/{nval}", flush=True)
    np.save(OUT / "validation_mse.npy", scores)
    counts_val = np.load(CORPUS / "validation/counts.npy")
    # Both global settings and settings calibrated to known synthetic count level.
    edges = [0, 10, 30, 100, 300, float("inf")]
    settings = {}
    for kind in ("gaussian", "fourier"):
        choices = np.array([i for i, c in enumerate(CANDIDATES) if c[0] == kind])
        settings[kind] = int(choices[np.argmin(scores[:, choices].mean(0))])
        settings[kind + "_by_counts"] = [
            int(
                choices[
                    np.argmin(
                        scores[(counts_val >= lo) & (counts_val < hi)][:, choices].mean(
                            0
                        )
                    )
                ]
            )
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    (OUT / "tuning.json").write_text(
        json.dumps(
            {
                "candidates": CANDIDATES,
                "selected_indices": settings,
                "count_bin_edges": ["0", "10", "30", "100", "300", "infinity"],
                "selection_metric": "mean validation MSE; 1000 validation examples; no test selection",
                "fourier_definition": "Even extension via orthonormal DCT-II; H=1/(1+((fy/cy)^2+(fx/cx)^2)^order), cutoff cycles/pixel; nonnegative clipping",
                "gaussian_definition": "Reflect boundary, sigma in pixels, axis order energy then momentum",
            },
            indent=2,
        )
    )
    print("Chosen settings:", settings, flush=True)
    model, _ = load_model(args.checkpoint, torch.device("cpu"))
    report = {}
    plot_samples = []
    for split in ("test", "ood"):
        ds = CorpusDataset(CORPUS, split, CONFIG["seed"])
        counts = np.load(CORPUS / split / "counts.npy")
        params = [
            json.loads(s)
            for s in (CORPUS / split / "parameters.jsonl").read_text().splitlines()
        ]
        names = [
            "input",
            "network",
            "gaussian",
            "fourier",
            "gaussian_by_counts",
            "fourier_by_counts",
        ]
        errors = {k: [] for k in names}
        ridge = {k: [] for k in names}
        peak = {k: [] for k in names}
        examples = [int(np.argmin(counts)), int(np.argsort(counts)[len(counts) // 2])]
        for i in range(len(ds)):
            x, clean, _, _ = ds[i]
            with torch.inference_mode():
                prediction = model(x[None])[0, 0].numpy()
            x, clean = x[0].numpy(), clean[0].numpy()
            results = {"input": x, "network": prediction}
            bin_index = int(np.searchsorted(edges, counts[i], side="right") - 1)
            for name in names[2:]:
                ix = (
                    settings[name][bin_index]
                    if name.endswith("_by_counts")
                    else settings[name]
                )
                results[name] = filtered(x, CANDIDATES[ix])
            # Dominant EDC peak only: not a multiband recovery metric.
            mask = clean.max(0) > 0.2 * clean.max()
            true_ridge = clean.argmax(0)
            delta_e = (params[i]["window"]["emax"] - params[i]["window"]["emin"]) / (
                clean.shape[0] - 1
            )
            for name, y in results.items():
                errors[name].append(float(np.mean((y - clean) ** 2)))
                ridge[name].append(
                    float(
                        np.mean(np.abs(y.argmax(0)[mask] - true_ridge[mask])) * delta_e
                    )
                )
                peak[name].append(
                    float(
                        np.mean(
                            np.abs(y.max(0)[mask] - clean.max(0)[mask])
                            / clean.max(0)[mask]
                        )
                    )
                )
            if i in examples:
                plot_samples.append((split, i, params[i], counts[i], clean, results))
            if (i + 1) % 100 == 0:
                print(f"{split}: {i+1}/{len(ds)}", flush=True)
        arrays = {k: np.asarray(v) for k, v in errors.items()}
        np.savez_compressed(
            OUT / f"{split}_per_sample.npz",
            counts=counts,
            **arrays,
            **{k + "_ridge_eV": v for k, v in ridge.items()},
            **{k + "_peak_relative_error": v for k, v in peak.items()},
        )
        report[split] = {
            "mse": summary(errors),
            "dominant_edc_peak_position_mae_eV": summary(ridge),
            "dominant_edc_peak_height_relative_mae": summary(peak),
            "count_bins": {},
            "families": {},
        }
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (counts >= lo) & (counts < hi)
            report[split]["count_bins"][f"{lo}-{hi}"] = {
                "n": int(mask.sum()),
                **{k: float(v[mask].mean()) for k, v in arrays.items()},
            }
        for family in sorted(set(p["family"] for p in params)):
            mask = np.array([p["family"] == family for p in params])
            report[split]["families"][family] = {
                k: float(v[mask].mean()) for k, v in arrays.items()
            }
        # Paired bootstrap over independent spectra, for strongest tuned baselines.
        rng = np.random.default_rng(981)
        report[split]["paired_mse_advantage"] = {}
        for name in ("gaussian_by_counts", "fourier_by_counts"):
            difference = arrays[name] - arrays["network"]
            boot = np.array(
                [
                    rng.choice(difference, len(difference), replace=True).mean()
                    for _ in range(2000)
                ]
            )
            report[split]["paired_mse_advantage"][name] = {
                "baseline_minus_network_mean": float(difference.mean()),
                "bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
                "network_win_fraction": float((difference > 0).mean()),
            }
    report["provenance"] = {
        "checkpoint_sha256": hashlib.sha256((args.checkpoint).read_bytes()).hexdigest(),
        "seed": CONFIG["seed"],
        "corpus": str(CORPUS),
        "note": "Fixed noise identical to original evaluation. Count-conditioned baselines use true synthetic noise level; not automatically available for experiment. Peak metrics compare to broadened clean target, not bare bands.",
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, split in zip(axes, ("test", "ood")):
        bins = report[split]["count_bins"]
        for name in ("input", "network", "gaussian_by_counts", "fourier_by_counts"):
            ax.plot(
                range(5),
                [v[name] for v in bins.values()],
                marker="o",
                label=name.replace("_by_counts", " (tuned by counts)"),
            )
        ax.set(
            yscale="log",
            title=split,
            xticks=range(5),
            xticklabels=["<10", "10–30", "30–100", "100–300", "≥300"],
            xlabel="Expected counts per mean pixel",
            ylabel="Mean squared error vs clean target",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "error_by_noise.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(plot_samples), 5, figsize=(17, 12), layout="constrained"
    )
    for row, (split, i, p, count, clean, results) in enumerate(plot_samples):
        for ax, name, y in zip(
            axes[row],
            ["Noisy", "Clean", "Network", "Gaussian", "Fourier"],
            [
                results["input"],
                clean,
                results["network"],
                results["gaussian_by_counts"],
                results["fourier_by_counts"],
            ],
        ):
            ax.imshow(
                y,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=0,
                vmax=np.percentile(clean, 99.5),
            )
            ax.set(title=name, xlabel="Momentum pixel", ylabel="Energy pixel")
        axes[row, 0].set_ylabel(
            f'{split} #{i}, {p["family"]}\n{count:.1f} counts/mean pixel'
        )
    fig.suptitle(
        "Prespecified examples: minimum and median counts in each test split; filters tuned on validation only"
    )
    fig.savefig(OUT / "comparisons.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps({s: report[s]["mse"] for s in ("test", "ood")}, indent=2), flush=True
    )


if __name__ == "__main__":
    main()
