"""Paired-checkpoint component fitting under fixed stronger backgrounds."""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from scipy.special import voigt_profile
from background_data import make_background_v2
from benchmark_overlap import components, E, SIGMA
from analyze_background import baseline, preservation_job


def stress(models, output, device, quantile, repeats=12, cases=None):
    jobs = []
    conditions = []
    cases = cases or ["single", "double_0.7", "weak_double", "broad_incoherent"]
    for ci, case in enumerate(cases):
        centers, widths, amps = components(case)
        signal = np.zeros((256, 256), dtype="float32")
        for c, g, a in zip(centers, widths, amps):
            signal += (
                a[None, :]
                * voigt_profile(E[:, None] - c[None, :], SIGMA, g[None, :])
                * 0.06
            )
        scale = float(signal.mean())
        signal /= scale
        true_areas = amps * 0.06 / scale
        for ri, regime in enumerate(["zero", "local_comparable", "local_buried"]):
            bg, meta = make_background_v2(
                signal, np.random.default_rng(720000 + ci * 3 + ri), regime
            )
            conditions.append(
                {"case": case, **meta, "total_mean_counts": 300, "repeats": repeats}
            )
            count_scale = 300 / float((signal + bg).mean())
            for repeat in range(repeats):
                rng = np.random.default_rng(730000 + ci * 36 + ri * 12 + repeat)
                x = (
                    rng.poisson((signal + bg).astype("float64") * count_scale)
                    / count_scale
                ).astype("float32")
                norm = float(x.mean())
                arrays = {
                    "raw": x,
                    "oracle_subtraction": x - bg,
                    "lower_envelope": x - baseline(x, quantile),
                }
                with torch.inference_mode():
                    for name, model in models.items():
                        pred = (
                            model(torch.from_numpy(x / norm)[None, None].to(device))[
                                0, 0
                            ]
                            .cpu()
                            .numpy()
                            * norm
                        )
                        arrays[name] = x - pred
                jobs.append((case, regime, repeat, arrays, centers, widths, true_areas))
    records = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        for i, rows in enumerate(pool.map(preservation_job, jobs), 1):
            records.extend(rows)
            if i % 24 == 0:
                print(f"Strong-background fitting: {i}/{len(jobs)}", flush=True)
    output = Path(output)
    (output / "strong-fit-conditions.json").write_text(
        json.dumps(conditions, indent=2) + "\n"
    )
    (output / "strong-fit-records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    summary = {}
    for case in cases:
        summary[case] = {}
        for regime in ["zero", "local_comparable", "local_buried"]:
            summary[case][regime] = {}
            for method in ["raw", "oracle_subtraction", "lower_envelope", "v1", "v2"]:
                rows = [
                    r
                    for r in records
                    if r["case"] == case
                    and r["background_kind"] == regime
                    and r["method"] == method
                ]
                summary[case][regime][method] = {
                    k: float(np.mean([r[k] for r in rows]))
                    for k in [
                        "center_mae_meV",
                        "width_mae_meV",
                        "area_relative_mae",
                        "fit_success",
                    ]
                }
    (output / "strong-fit-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
