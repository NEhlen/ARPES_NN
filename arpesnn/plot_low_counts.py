"""Plot the lowest-count held-out example from four distinct band families."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from apply_model import load_model
from train_tasks import CorpusDataset


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "arpesnn/models/training_v2_20260919"
    output = run / "examples_additional"
    output.mkdir(exist_ok=True)
    config = json.loads((run / "run_config.json").read_text())
    corpus = Path(config["corpus"])
    counts = np.load(corpus / "test/counts.npy")
    params = [
        json.loads(line)
        for line in (corpus / "test/parameters.jsonl").read_text().splitlines()
    ]
    selected, families = [], set()
    for i in np.argsort(counts):
        family = params[i]["family"]
        if family not in families:
            selected.append(int(i))
            families.add(family)
        if len(selected) == 4:
            break
    torch.set_num_threads(2)
    model, _ = load_model(run / "denoise/best_model.pt", torch.device("cpu"))
    dataset = CorpusDataset(corpus, "test", config["seed"])
    fig, axes = plt.subplots(4, 3, figsize=(12, 13), layout="constrained")
    records = []
    for row, i in enumerate(selected):
        x, clean, _, _ = dataset[i]
        with torch.inference_mode():
            prediction = model(x[None])[0, 0].numpy()
        clean = clean[0].numpy()
        noisy = x[0].numpy()
        snr = float(10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2)))
        w = params[i]["window"]
        extent = [w[k] for k in ("smin", "smax", "emin", "emax")]
        vmax = np.percentile(clean, 99.5)
        for ax, array, title in zip(
            axes[row],
            (noisy, clean, prediction),
            ("Noisy input", "Clean target", "Denoised"),
        ):
            ax.imshow(
                array,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            ax.set(title=title, xlabel="Cut coordinate (Å⁻¹)", ylabel="E − EF (eV)")
        axes[row, 0].set_title(
            f"{params[i]['family']}: {counts[i]:.2f} counts/mean pixel\nNoisy input; global SNR {snr:.1f} dB"
        )
        records.append(
            {
                "test_index": i,
                "family": params[i]["family"],
                "counts_per_mean_pixel": float(counts[i]),
                "global_snr_dB": snr,
            }
        )
    fig.suptitle(
        "Low-count held-out synthetic examples — same intensity scale within each row"
    )
    fig.savefig(output / "low_count_synthetic.png", dpi=140)
    plt.close(fig)
    statistics = {}
    for split in ("train", "test"):
        c = np.load(corpus / split / "counts.npy")
        statistics[split] = {
            "min": float(c.min()),
            "median": float(np.median(c)),
            "max": float(c.max()),
            "below_10": int((c < 10).sum()),
            "total": len(c),
        }
    (output / "noise_examples.json").write_text(
        json.dumps(
            {
                "counts_statistics": statistics,
                "examples": records,
                "selection": "Lowest counts in test, four distinct families",
                "snr_definition": "10 log10(mean(clean^2)/mean((noisy-clean)^2))",
            },
            indent=2,
        )
    )
    print(json.dumps(statistics))


if __name__ == "__main__":
    main()
