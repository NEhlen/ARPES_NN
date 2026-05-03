import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nn_pytorch import Unet

HEADER_PAIR_RE = re.compile(r"^\s*(\w+)\s*,\s*(\w+)\s*=\s*([^,]+)\s*,\s*([^,]+)\s*$")
DIM_RE = re.compile(r"^\s*dimX\s*,\s*dimY\s*=\s*(\d+)\s*,\s*(\d+)\s*$")


def parse_arpes_text(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    lines = path.read_text(errors="replace").replace("\r", "\n").splitlines()
    meta: dict[str, float | int] = {}
    data_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "#DAT":
            data_start = i + 1
            break

        dim_match = DIM_RE.match(stripped)
        if dim_match:
            meta["dimX"] = int(dim_match.group(1))
            meta["dimY"] = int(dim_match.group(2))
            continue

        pair_match = HEADER_PAIR_RE.match(stripped)
        if pair_match:
            meta[pair_match.group(1)] = float(pair_match.group(3))
            meta[pair_match.group(2)] = float(pair_match.group(4))

    if data_start is None:
        raise ValueError(f"{path} does not contain a #DAT section.")
    if "dimX" not in meta or "dimY" not in meta:
        raise ValueError(f"{path} does not define dimX and dimY.")

    values = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if stripped:
            values.append(float(stripped))

    dim_x = int(meta["dimX"])
    dim_y = int(meta["dimY"])
    expected_values = dim_x * dim_y
    if len(values) != expected_values:
        raise ValueError(
            f"{path} contains {len(values)} values, expected {expected_values}."
        )

    # ArpesBandmass writes the flattened data in x-major order and mirrors Y by
    # default. Match its loader: np.array(data).reshape((dimX, dimY)).T[::-1, :]
    return np.asarray(values, dtype=np.float32).reshape(dim_x, dim_y).T[::-1, :], meta


def normalize_for_model(data: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Input contains no finite values.")

    scale = float(np.nanmean(data[finite]))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[finite] = data[finite] / scale
    return normalized, scale


def resize_array(data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(data).float()[None, None, :, :]
    resized = F.interpolate(
        tensor,
        size=shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def coordinate_channels(
    meta: dict[str, float | int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    e_vals = np.linspace(
        float(meta.get("ymin", 0.0)),
        float(meta.get("ymax", rows)),
        rows,
        dtype=np.float32,
    )
    k_vals = np.linspace(
        float(meta.get("xmin", 0.0)),
        float(meta.get("xmax", cols)),
        cols,
        dtype=np.float32,
    )
    e_grid = np.repeat(e_vals[:, None], cols, axis=1)
    k_grid = np.repeat(k_vals[None, :], rows, axis=0)
    return e_grid, k_grid


def build_model_input(
    intensity: np.ndarray,
    meta: dict[str, float | int],
    input_shape: tuple[int, int],
    in_channels: int,
) -> np.ndarray:
    intensity_256 = resize_array(intensity, input_shape)
    if in_channels == 1:
        return intensity_256
    if in_channels != 3:
        raise ValueError(f"Unsupported checkpoint input channel count: {in_channels}.")

    e_grid, k_grid = coordinate_channels(meta, input_shape)
    return np.stack([intensity_256, e_grid, k_grid]).astype(np.float32)


def robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return float(np.nanmin(finite)), float(np.nanmax(finite))
    return float(vmin), float(vmax)


def plot_prediction(
    output_path: Path,
    raw_data: np.ndarray,
    input_256: np.ndarray,
    prediction_256: np.ndarray,
    title: str,
    meta: dict[str, float | int],
) -> None:
    extent = [
        float(meta.get("xmin", 0.0)),
        float(meta.get("xmax", raw_data.shape[1])),
        float(meta.get("ymin", 0.0)),
        float(meta.get("ymax", raw_data.shape[0])),
    ]
    pred_extent = [extent[0], extent[1], extent[2], extent[3]]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = [
        ("raw input", raw_data, extent),
        ("model input", input_256, pred_extent),
        ("model output", prediction_256, pred_extent),
    ]

    for ax, (label, data, image_extent) in zip(axes, panels, strict=True):
        vmin, vmax = robust_limits(data)
        image = ax.imshow(
            data,
            aspect="auto",
            extent=image_extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(label)
        ax.set_xlabel("k")
        ax.set_ylabel("E")
        ax.invert_yaxis()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Unet, int]:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    in_channels = int(state_dict["conv1.weight"].shape[1])
    model = Unet(in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, in_channels


def apply_model_to_file(
    model: Unet,
    path: Path,
    output_dir: Path,
    device: torch.device,
    input_shape: tuple[int, int],
    in_channels: int,
) -> dict[str, str | float | int]:
    raw_data, meta = parse_arpes_text(path)
    normalized, normalization_scale = normalize_for_model(raw_data)
    model_input = build_model_input(normalized, meta, input_shape, in_channels)
    intensity_256 = model_input if in_channels == 1 else model_input[0]

    with torch.no_grad():
        if in_channels == 1:
            tensor = torch.from_numpy(model_input).float()[None, None, :, :].to(device)
        else:
            tensor = torch.from_numpy(model_input).float()[None, :, :, :].to(device)
        prediction_256 = model(tensor)[0, 0].cpu().numpy()

    prediction_original_grid = resize_array(prediction_256, raw_data.shape)
    output_prefix = output_dir / path.stem
    np.save(output_prefix.with_name(f"{output_prefix.name}_input_256.npy"), model_input)
    np.save(
        output_prefix.with_name(f"{output_prefix.name}_prediction_256.npy"),
        prediction_256,
    )
    np.savetxt(
        output_prefix.with_name(f"{output_prefix.name}_prediction_256.txt"),
        prediction_256,
    )
    np.savetxt(
        output_prefix.with_name(f"{output_prefix.name}_prediction_original_grid.txt"),
        prediction_original_grid,
    )
    plot_prediction(
        output_prefix.with_name(f"{output_prefix.name}_model_output.png"),
        normalized,
        intensity_256,
        prediction_256,
        path.name,
        meta,
    )

    return {
        "file": path.name,
        "dim_x": int(meta["dimX"]),
        "dim_y": int(meta["dimY"]),
        "finite_fraction": float(np.isfinite(raw_data).mean()),
        "normalization_scale": normalization_scale,
        "input_channels": in_channels,
        "input_256_min": float(np.nanmin(intensity_256)),
        "input_256_max": float(np.nanmax(intensity_256)),
        "prediction_256_min": float(np.nanmin(prediction_256)),
        "prediction_256_max": float(np.nanmax(prediction_256)),
        "png": f"{path.stem}_model_output.png",
    }


def write_overview(output_dir: Path, rows: list[dict[str, str | float | int]]) -> None:
    image_paths = [output_dir / str(row["png"]) for row in rows]
    if not image_paths:
        return

    fig, axes = plt.subplots(
        len(image_paths),
        1,
        figsize=(13.5, 4.0 * len(image_paths)),
        constrained_layout=True,
    )
    if len(image_paths) == 1:
        axes = [axes]

    for ax, image_path in zip(axes, image_paths, strict=True):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.axis("off")

    fig.savefig(output_dir / "overview.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained ARPES U-Net checkpoint to ARPES text files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../ArpesBandmass/temp"),
        help="Directory containing ARPES text files with #DAT payloads.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("arpesnn/models/best_model.pt"),
        help="Path to a saved U-Net state_dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("arpesnn/models/example_outputs"),
        help="Directory for PNGs and numeric predictions.",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for files under --input-dir.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_channels = load_model(args.checkpoint, device)

    rows = []
    for path in paths:
        rows.append(
            apply_model_to_file(
                model=model,
                path=path,
                output_dir=output_dir,
                device=device,
                input_shape=(256, 256),
                in_channels=in_channels,
            )
        )
        print(f"wrote prediction for {path.name}")

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_overview(output_dir, rows)
    print(f"wrote summary to {summary_path}")
    print(f"wrote overview to {output_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
