"""Read SPECS P2 image blocks, preserving acquisition metadata and native axes.

SPECS stores energy along image columns and angle along rows (also used by
NavARP's load_specs_sp2). We return intensity[energy, angle] without flipping.
Reference: https://gitlab.com/fbisti/navarp/-/blob/master/navarp/utils/navfile.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SP2Image:
    intensity: np.ndarray
    energy_eV: np.ndarray | None
    angle_deg: np.ndarray | None
    metadata: dict[str, str]
    index: int


def load_sp2(path: str | Path) -> list[SP2Image]:
    """Load all P2 blocks; no count rescaling or angle-to-momentum conversion."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    blocks = []
    metadata = {}
    tokens = []
    started = False

    def finish():
        if len(tokens) < 3:
            raise ValueError("Truncated P2 header")
        try:
            width, height, maximum = map(int, tokens[:3])
            values = np.array(tokens[3:], dtype=np.float64)
        except ValueError as error:
            raise ValueError("Invalid numeric P2 payload") from error
        if width <= 0 or height <= 0 or maximum <= 0:
            raise ValueError("P2 dimensions and maximum must be positive")
        if values.size != width * height:
            raise ValueError(
                f"P2 block {len(blocks)}: expected {width * height} pixels, got {values.size}"
            )
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values > maximum).any()
        ):
            raise ValueError("P2 pixels must be finite and within the declared range")
        try:
            erange = [float(v) for v in metadata["ERange"].split()]
            arange = [float(v) for v in metadata["aRange"].split()]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "SP2 requires numeric ERange and aRange metadata"
            ) from error
        if (
            len(erange) != 2
            or len(arange) != 2
            or not np.isfinite(erange + arange).all()
        ):
            raise ValueError("SP2 axis ranges must contain two finite endpoints")
        if metadata.get("aUnit", "deg").strip('"') != "deg":
            raise ValueError("Only angular SP2 axes in degrees are supported")
        labels = metadata.get("Images", "Corrected").strip('"').split()
        label = labels[len(blocks)] if len(blocks) < len(labels) else "Unknown"
        calibrated = label.lower() == "corrected"
        block_metadata = dict(
            metadata,
            image_label=label,
            axis_calibration=(
                "header_linear_ranges" if calibrated else "detector_pixels_only"
            ),
        )
        blocks.append(
            SP2Image(
                values.reshape(height, width).T.copy(),
                np.linspace(*erange, width) if calibrated else None,
                np.linspace(*arange, height) if calibrated else None,
                block_metadata,
                len(blocks),
            )
        )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "P2":
            if started:
                finish()
            started = True
            tokens = []
        elif line.startswith("#"):
            if "=" in line:
                key, value = line[1:].split("=", 1)
                metadata[key.strip()] = value.split("#", 1)[0].strip()
        else:
            if not started:
                raise ValueError("Expected P2 magic header")
            tokens.extend(line.split("#", 1)[0].split())
    if not started:
        raise ValueError("No P2 image blocks found")
    finish()
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    # Fail before writing if any input or output is ambiguous.
    loaded = [(p, load_sp2(p)) for p in args.paths]
    stems = [p.stem for p in args.paths]
    if len(stems) != len(set(stems)):
        parser.error("Input basenames must be unique")
    for p, blocks in loaded:
        for block in blocks:
            for suffix in (".npz", ".png"):
                if (args.output_dir / f"{p.stem}_block{block.index}{suffix}").exists():
                    parser.error("Output exists; use a fresh output directory")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for p, blocks in loaded:
        for block in blocks:
            prefix = args.output_dir / f"{p.stem}_block{block.index}"
            calibrated = block.energy_eV is not None
            axes = (
                {"energy_eV": block.energy_eV, "angle_deg": block.angle_deg}
                if calibrated
                else {
                    "detector_column": np.arange(block.intensity.shape[0]),
                    "detector_row": np.arange(block.intensity.shape[1]),
                }
            )
            np.savez_compressed(
                prefix.with_suffix(".npz"),
                intensity=block.intensity,
                metadata_json=json.dumps(block.metadata),
                axis_order=(
                    "energy,angle" if calibrated else "detector_column,detector_row"
                ),
                **axes,
            )
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            x = block.angle_deg if calibrated else axes["detector_row"]
            y = block.energy_eV if calibrated else axes["detector_column"]
            image = ax.pcolormesh(
                x,
                y,
                block.intensity,
                shading="auto",
                cmap="magma",
                vmax=np.percentile(block.intensity, 99.5),
            )
            ax.set(
                xlabel="Emission angle (deg)" if calibrated else "Detector row (pixel)",
                ylabel=(
                    "Kinetic energy (eV)" if calibrated else "Detector column (pixel)"
                ),
                title=f"{p.name} | {block.metadata['image_label']} (block {block.index})",
            )
            fig.colorbar(image, ax=ax, label="Stored intensity (uncalibrated units)")
            fig.savefig(prefix.with_suffix(".png"), dpi=140)
            plt.close(fig)
        print(
            f"{p.name}: {len(blocks)} blocks, energy x angle = {blocks[0].intensity.shape}"
        )


if __name__ == "__main__":
    main()
