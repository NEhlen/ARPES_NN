"""Background figures with explicit, separate signal and background scales."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def background_limit(*arrays):
    # Include the full estimate, including false positives on zero-background controls.
    return max(float(np.max(a)) for a in arrays) or 1.0


def plot_examples(items, path):
    fig, axes = plt.subplots(
        len(items),
        5,
        figsize=(20, 3.4 * len(items)),
        layout="constrained",
        squeeze=False,
    )
    for row, (title, signal, x, bg, pred, corrected) in enumerate(items):
        hi = np.percentile(x, 99.5)
        bg_hi = background_limit(bg, pred)
        images = []
        for col, (array, name) in enumerate(
            zip(
                [x, bg, pred, signal, corrected],
                [
                    "Noisy input",
                    "True background",
                    "Estimated background",
                    "True signal",
                    "Input − estimate",
                ],
            )
        ):
            is_bg = col in (1, 2)
            images.append(
                axes[row, col].imshow(
                    array,
                    origin="lower",
                    aspect="auto",
                    cmap="viridis" if is_bg else "magma",
                    vmin=0,
                    vmax=bg_hi if is_bg else hi,
                )
            )
            axes[row, col].set(title=name, xlabel="Momentum pixel")
        axes[row, 0].set_ylabel(title + "\nEnergy pixel")
        fig.colorbar(
            images[2],
            ax=list(axes[row, 1:3]),
            shrink=0.8,
            label="Background scale (same intensity units)",
        )
        fig.colorbar(
            images[4],
            ax=[axes[row, i] for i in (0, 3, 4)],
            shrink=0.8,
            label="Signal scale (intensity units)",
        )
    fig.suptitle(
        "Separate linear scales per row: backgrounds share viridis; spectra share magma\n"
        "Negative corrected values display black; signed arrays are retained"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)
