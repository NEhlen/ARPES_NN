"""Compare two checkpoints on identical old/new draws, grouped by contrast regime."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from background_data import BackgroundDataset, local_band_mask
from background_model import load_background
from background_plots import plot_examples
from analyze_background import baseline


def compare(corpus, old, new, output, device="cpu", skip_stress=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    models = {
        name: load_background(path, device)[0]
        for name, path in [("v1", old), ("v2", new)]
    }
    summaries = {}
    for profile in ["v1", "v2"]:
        # Tune the conventional baseline independently for each acquisition distribution.
        scores = np.zeros(4)
        qs = [0.02, 0.05, 0.1, 0.2]
        for item in BackgroundDataset(corpus, "validation", 42, profile):
            x, b = item["input"][0].numpy(), item["background"][0].numpy()
            scores += [np.mean((baseline(x, q) - b) ** 2) for q in qs]
        q = qs[int(scores.argmin())]
        summaries[profile] = {"baseline_quantile": q, "splits": {}}
        for split in ["test", "ood"]:
            records, examples, selected = [], {"v1": [], "v2": []}, set()
            for batch in DataLoader(
                BackgroundDataset(corpus, split, 42, profile), batch_size=32
            ):
                with torch.inference_mode():
                    predictions = {
                        name: model(batch["input"].to(device)).cpu().numpy()[:, 0]
                        for name, model in models.items()
                    }
                for j in range(len(batch["input"])):
                    x, s, b = [
                        batch[key][j, 0].numpy()
                        for key in ["input", "signal", "background"]
                    ]
                    regime = batch["regime"][j]
                    mask = local_band_mask(s)
                    estimates = {
                        name: values[j] for name, values in predictions.items()
                    }
                    estimates.update(
                        none=np.zeros_like(b), lower_envelope=baseline(x, q)
                    )
                    for method, p in estimates.items():
                        error = p - b
                        records.append(
                            dict(
                                index=int(batch["index"][j]),
                                regime=regime,
                                method=method,
                                mean_background_signal=float(
                                    batch["background_to_signal_mean"][j]
                                ),
                                local_signal_background=float(
                                    batch["local_signal_to_background"][j]
                                ),
                                background_mse=float(np.mean(error**2)),
                                ridge_error_relative_rms=float(
                                    np.sqrt(
                                        np.mean(error[mask] ** 2)
                                        / np.mean(s[mask] ** 2)
                                    )
                                ),
                                absolute_area_error=float(abs(error.mean()) / s.mean()),
                                ridge_signed_bias=float(
                                    error[mask].mean() / s[mask].mean()
                                ),
                            )
                        )
                    if regime not in selected:
                        for name in models:
                            p = estimates[name]
                            examples[name].append(
                                (
                                    f"{regime}: B/S={float(batch['background_to_signal_mean'][j]):.2g}",
                                    s,
                                    x,
                                    b,
                                    p,
                                    x - p,
                                )
                            )
                        selected.add(regime)
            (output / f"{profile}-{split}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n"
            )
            groups = {}
            for regime in sorted(selected | {"all"}):
                groups[regime] = {}
                for method in ["none", "lower_envelope", "v1", "v2"]:
                    rows = [
                        r
                        for r in records
                        if r["method"] == method
                        and (regime == "all" or r["regime"] == regime)
                    ]
                    groups[regime][method] = {"n": len(rows)}
                    for metric in [
                        "background_mse",
                        "ridge_error_relative_rms",
                        "absolute_area_error",
                        "ridge_signed_bias",
                    ]:
                        values = [r[metric] for r in rows]
                        groups[regime][method][metric] = float(np.mean(values))
                        groups[regime][method][metric + "_p90"] = float(
                            np.percentile(values, 90)
                        )
            summaries[profile]["splits"][split] = groups
            if profile == "v2":
                for name in models:
                    plot_examples(examples[name], output / f"{split}-{name}.png")
            print(f"Compared {profile}/{split}", flush=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Background v2 comparison",
        "",
        "Identical held-out draws per model. Local ridge error measures error in estimated background divided by true signal RMS in a signal-defined ridge neighbourhood. It isolates subtraction distortion from shot noise; it is not a fitted band-position metric.",
        "",
        "| Profile / split / regime | V1 ridge error | V2 ridge error | V1 area error | V2 area error |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile, value in summaries.items():
        for split, groups in value["splits"].items():
            for regime, metrics in groups.items():
                a, b = metrics["v1"], metrics["v2"]
                lines.append(
                    f"| {profile}/{split}/{regime} | {a['ridge_error_relative_rms']:.3f} | {b['ridge_error_relative_rms']:.3f} | {a['absolute_area_error']:.3f} | {b['absolute_area_error']:.3f} |"
                )
    lines += [
        "",
        "Examples are the first encountered member of each regime, not selected for performance. Scales differ between model figures; use the colorbars. All summary groups and 90th percentiles are in summary.json. No experimental accuracy claim follows from this benchmark.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    from stress_background_v2 import stress

    if not skip_stress:
        stress(models, output, device, summaries["v2"]["baseline_quantile"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["corpus", "old", "new", "output"]:
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--skip-stress",
        action="store_true",
        help="Only image/regime metrics; for smoke checks",
    )
    compare(**vars(p.parse_args()))
