# Publication notes

The repository is prepared locally; no commit, push, GitHub release or visibility change is performed by the preparation workflow.

## Included in the source repository

Code, tests, dependency lockfile, README, model/training/benchmark documentation, curated PNG figures and compact numeric benchmark summaries. The older tracked `example_dataset` is a small historical bare-band example, explicitly labelled in the README.

## Kept local

`.env`, virtual environments, generated corpora, training checkpoints, raw files under `data/`, labbook scans, output folders and release archives are ignored. Ignoring a path does not remove already tracked files; review `git status` and `git ls-files` before publishing. Existing data is preserved on disk.

## License and optional distribution

1. **License: MIT.** The repository includes `LICENSE`; the prepared inference-weight archive includes the same license.
2. Optionally distribute the pretrained denoiser as a GitHub Release asset, rather than committing training checkpoints. Until an asset is actually published, the README intentionally supplies training commands rather than a fabricated download link.

## Prepare an optional inference-weight archive

```bash
uv run python scripts/prepare_release.py \
  --checkpoint arpesnn/models/training_v2_20260919/denoise/best_model.pt \
  --output releases/arpes-denoiser-v1
```

This creates an inspectable folder and `.tar.gz` containing inference weights, a model card, portable model metadata and SHA-256 checksums. It excludes optimizer state, raw data, corpora and machine-specific training metadata. It does not upload anything. The MIT license is included in the prepared archive.

## Local verification

```bash
uv sync --locked --extra dev
uv run python -m unittest discover -s tests -v
uv run black --check arpesnn tests scripts
```

Also run the README demo and smoke training/inference steps in a clean copy without `data/` or local models. Inspect Markdown image links from that copy. A passing local run does not claim that GitHub Actions itself has run.
