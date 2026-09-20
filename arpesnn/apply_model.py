import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nn_pytorch import Unet
from compact_model import CompactUNet
from sp2 import load_sp2

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


def load_experimental_file(path: Path, sp2_block: int = 0):
    if path.suffix.lower() != ".sp2":
        return parse_arpes_text(path)
    blocks = load_sp2(path)
    if not 0 <= sp2_block < len(blocks):
        raise ValueError(
            f"SP2 block {sp2_block} unavailable; file contains {len(blocks)} blocks"
        )
    block = blocks[sp2_block]
    if block.energy_eV is None or block.angle_deg is None:
        raise ValueError(
            "Raw detector blocks need instrument correction before inference; select the corrected block."
        )
    meta = {
        "dimX": block.intensity.shape[1],
        "dimY": block.intensity.shape[0],
        "xmin": float(block.angle_deg[0]),
        "xmax": float(block.angle_deg[-1]),
        "ymin": float(block.energy_eV[0]),
        "ymax": float(block.energy_eV[-1]),
        "x_label": "Emission angle (deg)",
        "y_label": "Kinetic energy (eV)",
        "coordinate_system": "kinetic_energy_angle",
        "sp2_block": sp2_block,
        "acquisition": block.metadata,
    }
    return block.intensity.astype(np.float32), meta


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
    intensity_model_grid = resize_array(intensity, input_shape)
    if in_channels == 1:
        return intensity_model_grid
    if in_channels != 3:
        raise ValueError(f"Unsupported checkpoint input channel count: {in_channels}.")

    if meta.get("coordinate_system") == "kinetic_energy_angle":
        raise ValueError(
            "A 3-channel E,k model cannot use SP2 kinetic energy/angle directly. Calibrate EF and convert angle to momentum first, or use an intensity-only model."
        )
    e_grid, k_grid = coordinate_channels(meta, input_shape)
    return np.stack([intensity_model_grid, e_grid, k_grid]).astype(np.float32)


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
    input_model_grid: np.ndarray,
    prediction_model_grid: np.ndarray,
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
        ("model input", input_model_grid, pred_extent),
        ("model output", prediction_model_grid, pred_extent),
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
        ax.set_xlabel(str(meta.get("x_label", "k")))
        ax.set_ylabel(str(meta.get("y_label", "E")))
        if meta.get("coordinate_system") != "kinetic_energy_angle":
            ax.invert_yaxis()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, int]:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    if state_dict.get("architecture") == "compact_unet_v1":
        model = CompactUNet(state_dict["task"], state_dict["base_channels"]).to(device)
        model.load_state_dict(state_dict["state_dict"])
        model.eval()
        return model, 1
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
    sp2_block: int = 0,
    checkpoint_task: str = "unknown",
) -> dict[str, str | float | int]:
    raw_data, meta = load_experimental_file(path, sp2_block)
    normalized, normalization_scale = normalize_for_model(raw_data)
    model_input = build_model_input(normalized, meta, input_shape, in_channels)
    intensity_model_grid = model_input if in_channels == 1 else model_input[0]

    with torch.no_grad():
        if in_channels == 1:
            tensor = torch.from_numpy(model_input).float()[None, None, :, :].to(device)
        else:
            tensor = torch.from_numpy(model_input).float()[None, :, :, :].to(device)
        prediction_model_grid = model(tensor)[0, 0].cpu().numpy()

    prediction_original_grid = resize_array(prediction_model_grid, raw_data.shape)
    output_prefix = output_dir / path.stem
    with output_prefix.with_name(f"{output_prefix.name}_metadata.json").open(
        "w"
    ) as handle:
        json.dump(
            dict(
                meta,
                normalization_scale=normalization_scale,
                output_units="normalized_training_target",
                checkpoint_task=checkpoint_task,
                resampled_shape=list(input_shape),
            ),
            handle,
            indent=2,
        )
    np.save(
        output_prefix.with_name(f"{output_prefix.name}_input_model_grid.npy"),
        model_input,
    )
    np.save(
        output_prefix.with_name(f"{output_prefix.name}_prediction_model_grid.npy"),
        prediction_model_grid,
    )
    np.savetxt(
        output_prefix.with_name(f"{output_prefix.name}_prediction_model_grid.txt"),
        prediction_model_grid,
    )
    np.savetxt(
        output_prefix.with_name(f"{output_prefix.name}_prediction_original_grid.txt"),
        prediction_original_grid,
    )
    plot_prediction(
        output_prefix.with_name(f"{output_prefix.name}_model_output.png"),
        normalized,
        intensity_model_grid,
        prediction_model_grid,
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
        "input_model_grid_min": float(np.nanmin(intensity_model_grid)),
        "input_model_grid_max": float(np.nanmax(intensity_model_grid)),
        "prediction_model_grid_min": float(np.nanmin(prediction_model_grid)),
        "prediction_model_grid_max": float(np.nanmax(prediction_model_grid)),
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
        default=Path("data/experiment_data/Elettra-Feb18/09"),
        help="Directory containing SPECS SP2 files or ARPES text files with #DAT payloads.",
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
        default="*.sp2",
        help="Glob pattern for files under --input-dir.",
    )
    parser.add_argument(
        "--sp2-block",
        type=int,
        default=0,
        help="Zero-based P2 block; default is the first (corrected) image.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input_dir / args.pattern}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_channels = load_model(args.checkpoint, device)
    metadata_path = args.checkpoint.with_suffix(".metadata.json")
    input_shape = (256, 256)
    checkpoint_task = "unknown"
    if metadata_path.exists():
        with metadata_path.open() as handle:
            checkpoint_metadata = json.load(handle)
        input_shape = tuple(checkpoint_metadata["input_shape"])
        checkpoint_task = checkpoint_metadata["task"]
        print(f"Checkpoint task: {checkpoint_metadata['task']}")

    rows = []
    for path in paths:
        rows.append(
            apply_model_to_file(
                model=model,
                path=path,
                output_dir=output_dir,
                device=device,
                input_shape=input_shape,
                in_channels=in_channels,
                sp2_block=args.sp2_block,
                checkpoint_task=checkpoint_task,
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
