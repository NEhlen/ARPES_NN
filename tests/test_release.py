"""Portable task-specific release bundles must preserve inference weights."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arpesnn"))
from compact_model import CompactUNet
from background_model import BackgroundNet, load_background


class ReleaseTests(unittest.TestCase):
    def test_both_task_archives_and_existing_output_protection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task, architecture, model, name in [
                (
                    "denoise",
                    "compact_unet_v1",
                    CompactUNet("denoise", 16),
                    "denoiser.pt",
                ),
                (
                    "background",
                    "background_coarse_v1",
                    BackgroundNet(),
                    "background.pt",
                ),
            ]:
                checkpoint = root / f"{task}.pt"
                torch.save(
                    {
                        "architecture": architecture,
                        "task": task,
                        "base_channels": 16,
                        "input_shape": [32, 32],
                        "profile": "v2" if task == "background" else "reference",
                        "state_dict": model.state_dict(),
                    },
                    checkpoint,
                )
                folder = root / task
                command = [
                    sys.executable,
                    str(ROOT / "scripts/prepare_release.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(folder),
                ]
                subprocess.run(command, check=True, capture_output=True)
                self.assertEqual(checkpoint.read_bytes(), (folder / name).read_bytes())
                metadata = json.loads((folder / "model.json").read_text())
                self.assertEqual(metadata["task"], task)
                self.assertEqual(
                    metadata["checkpoint_sha256"],
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                )
                for line in (folder / "SHA256SUMS.txt").read_text().splitlines():
                    digest, filename = line.split("  ", 1)
                    self.assertEqual(
                        hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                        digest,
                    )
                with tarfile.open(folder.with_suffix(".tar.gz")) as archive:
                    self.assertEqual(
                        {Path(m.name).name for m in archive if m.isfile()},
                        {
                            name,
                            "model.json",
                            "MODEL_CARD.md",
                            "README.txt",
                            "LICENSE",
                            "SHA256SUMS.txt",
                        },
                    )
                if task == "background":
                    loaded, _ = load_background(folder / name, "cpu")
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            loaded(torch.ones(1, 1, 32, 32)),
                            model.eval()(torch.ones(1, 1, 32, 32)),
                        )
                before = (folder.with_suffix(".tar.gz")).read_bytes()
                self.assertNotEqual(
                    subprocess.run(command, capture_output=True).returncode, 0
                )
                self.assertEqual(before, (folder.with_suffix(".tar.gz")).read_bytes())


if __name__ == "__main__":
    unittest.main()
