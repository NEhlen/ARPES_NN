"""Reproduce shared-scale synthetic and experimental prediction comparisons."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from apply_model import load_model, load_experimental_file, resize_array
from train_tasks import save_prediction_preview

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "arpesnn/models/training_v2_20260919"
OUT = RUN / "examples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", nargs="+", default=["067", "074"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    models = {
        task: load_model(RUN / task / "best_model.pt", device)[0]
        for task in ("denoise", "bareband")
    }
    config = json.loads((RUN / "run_config.json").read_text())
    save_prediction_preview(
        models,
        config["corpus"],
        config["seed"],
        device,
        output / "synthetic_comparison.png",
    )
    records = []
    for number in args.numbers:
        source = ROOT / f"data/experiment_data/Elettra-Feb18/09/mos2_2_{number}.sp2"
        raw, meta = load_experimental_file(source)
        model_input = resize_array(raw, (256, 256))
        scale = float(model_input.mean())
        with torch.inference_mode():
            pred = (
                models["denoise"](torch.from_numpy(model_input / scale)[None, None])[
                    0, 0
                ].numpy()
                * scale
            )
        residual = model_input - pred
        extent = [meta[k] for k in ("xmin", "xmax", "ymin", "ymax")]
        lo, hi = np.percentile(model_input, [1, 99.5])
        limit = max(float(np.percentile(np.abs(residual), 99)), 1e-8)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), layout="constrained")
        titles = [
            f"Measured ({raw.shape[1]} × {raw.shape[0]})",
            "Resampled input (256 × 256)",
            "Denoiser output (256 × 256)",
            "Input − output",
        ]
        for i, (ax, data, title) in enumerate(
            zip(axes, (raw, model_input, pred, residual), titles)
        ):
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="RdBu_r" if i == 3 else "magma",
                vmin=-limit if i == 3 else lo,
                vmax=limit if i == 3 else hi,
            )
            ax.set(
                title=title, xlabel="Emission angle (deg)", ylabel="Kinetic energy (eV)"
            )
            fig.colorbar(im, ax=ax, shrink=0.8, label="Recorded intensity units")
        fig.suptitle(
            f"{source.name} — experimental inference, no clean reference\n"
            "First three panels share one intensity scale; resampling itself also smooths the data",
            fontsize=13,
        )
        fig.savefig(output / f"real_{number}.png", dpi=150)
        plt.close(fig)
        np.savez_compressed(
            output / f"real_{number}.npz",
            raw=raw,
            model_input=model_input,
            prediction=pred,
            residual=residual,
            extent=extent,
        )
        records.append(
            {
                "source": str(source),
                "checkpoint": str(RUN / "denoise/best_model.pt"),
                "normalization_mean": scale,
                "metadata": meta,
                "note": "Bilinear resampling before inference; output rescaled to input intensity units. No experimental ground truth.",
            }
        )
    (output / "provenance.json").write_text(json.dumps(records, indent=2))
    print(output)


if __name__ == "__main__":
    main()
