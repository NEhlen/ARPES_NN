# Repository Guidelines

## Project Focus

- Prioritize **ARPES denoising**: remove counting noise while preserving physical broadening, renormalization, instrumental resolution and line shapes.
- This is an exploratory, MIT-licensed project intended to be understandable and usable from GitHub. It is not a validated self-energy extraction method or a paper implementation.
- Retain the older bare-band/self-energy-removal workflows for compatibility, but do not present them as the recommended path or confuse their targets with denoising targets.
- Use `README.md` for the public quickstart, `docs/TRAINING.md` for simulation/training, `docs/MODEL_CARD.md` for model assumptions, and `docs/BENCHMARKS.md` for evaluation methods and limitations.

## Project Structure

- `arpesnn/demo.py`: deterministic synthetic example, optionally with checkpoint inference.
- `arpesnn/denoise.py`: public inference CLI for nonnegative 2D NumPy arrays and corrected SP2 images.
- `arpesnn/corpus.py`, `band_models.py`: synthetic configurations, spectral intensities and disk-backed corpus generation.
- `arpesnn/compact_model.py`, `train_tasks.py`: compact residual U-Net and training; explicitly use `--task denoise` for denoising-only runs.
- `arpesnn/sp2.py`: SP2 reader. Raw detector blocks require instrument correction; do not invent energy, Fermi-level or momentum calibration.
- `arpesnn/benchmark_denoising.py`: validation-tuned Gaussian/Fourier comparison against clean synthetic intensity.
- `arpesnn/benchmark_overlap.py`, `report_overlap.py`: repeated-noise component-fitting benchmark and reporting.
- `arpesnn/audit_corpus.py`: generated-data integrity and reproducibility checks.
- `docs/figures/`, `docs/results/`: curated, publishable figures and compact numeric results; figure provenance belongs in `docs/FIGURES.md`.
- `tests/`: CPU-compatible regression tests using `unittest`; `.github/workflows/tests.yml` also runs a small training/inference smoke test.
- `scripts/prepare_release.py`: prepares an inference-weight archive without uploading it.
- `generate_data.py`, `generate_tasks.py`, `nn_pytorch.py` and `apply_model.py` are legacy/pilot workflows. The tracked `arpesnn/dataset/example_dataset/` is a legacy **bare-band** example.

## Setup and Commands

Run scripts from the repository root. Python 3.12 is the reference and CI version; dependencies are managed with `uv`. The project currently exposes scripts, not an installed package API.

```bash
uv sync --locked --extra dev
uv run python arpesnn/demo.py --output outputs/demo
uv run python -m unittest discover -s tests -v
uv run black --check arpesnn tests scripts
```

For a small pipeline check, use fresh output directories:

```bash
uv run python arpesnn/corpus.py --output arpesnn/dataset/smoke \
  --train 40 --validation 10 --test 10 --ood 10 --size 64 --workers 2
uv run python arpesnn/train_tasks.py --corpus arpesnn/dataset/smoke \
  --output arpesnn/models/smoke --task denoise \
  --epochs 2 --batch-size 8 --workers 0
uv run python arpesnn/denoise.py --input outputs/demo/noisy.npy \
  --checkpoint arpesnn/models/smoke/denoise/best_model.pt \
  --size 64 --device cpu --output outputs/smoke-prediction
```

- A smoke model is not a scientifically useful pretrained model. The reference model was trained at 256 × 256; inference `--size` must reflect the intended model grid.
- Use `--require-cuda` for long GPU training runs to avoid silently falling back to CPU. Verify device access in the execution environment; sandbox visibility can differ from the host.
- Do not launch full corpus generation, training or expensive benchmarks just to verify documentation changes.
- Modern workflows take explicit CLI paths and do not need `.env`. `DATASET_PATH` and `MODELS_PATH` are legacy configuration options.
- When changing dependencies, update `pyproject.toml` and `uv.lock` together.

## Coding and Verification

- Use four-space indentation, PEP 8 naming, lowercase module names, `CamelCase` classes and `snake_case` functions/variables. Format changed Python files with Black.
- Add meaningful regression tests under `tests/test_*.py`, following the existing `unittest` suite. No separate pytest dependency is required.
- Tests must not depend on private measurements, local trained checkpoints or a GPU. Generate small fixtures; skip optional local-data integration tests when those data are absent.
- Verify user-facing commands from a clean copy without private data or generated models when changing setup, training or inference behavior. Distinguish local verification from a GitHub Actions run.
- Preserve seed reproducibility, split isolation, parameter/source provenance and checkpoint compatibility. Avoid loading generated artifacts at module import time.
- Public scripts should accept explicit paths rather than assume the original author's machine or historical training directory.

## Scientific and Reporting Conventions

- Inputs use energy rows and momentum/angle columns. The public `denoise.py` CLI restores input intensity units on the model grid; the older `apply_model.py` exports normalized target units. Keep these contracts explicit.
- Show the resampled input separately from the original acquisition. Interpolating predictions back to acquisition dimensions does not recover lost resolution.
- Use matched intensity scales for comparisons, label residual scales and document example selection. Do not cherry-pick only favorable predictions.
- Keep clean-intensity MSE, dominant-pixel position, fitted component centers and bare-band energies conceptually separate.
- Tune methods on calibration/validation data, never the final test set. Group correlated EDCs by noise realization when bootstrapping uncertainty.
- Report fit bias, linewidth changes, false splitting and failures alongside improvements. Ordinary independent-pixel fit covariance is not calibrated uncertainty after denoising.
- Held-out band families still share simulator assumptions. Synthetic success and a smoother experimental image do not establish experimental accuracy.
- Keep limitations visible: Poisson-only training noise, domain/pixel-scale mismatch, weak-feature distortion, ill-conditioned overlaps and possible apparent gaps from forced multi-component fits. The broad-component test is not a complete microscopic waterfall model.

## Data, Licensing and Releases

- Keep `data/` (including labbooks), generated corpora/checkpoints, `.env`, `outputs/` and `releases/` out of Git. Preserve local files; do not delete them as repository cleanup.
- The existing tracked legacy example is an explicit exception. Check `git ls-files` as well as ignore rules before publication; `.gitignore` does not untrack files.
- Include only deliberately curated figures/results in documentation, with provenance. Do not add raw experimental data or labbook scans by default.
- The project uses **MIT**; retain `LICENSE` and keep package, documentation and release licensing consistent.
- Distribute the trained denoiser as a **GitHub Release attachment**, not a Git-tracked checkpoint. Keep the bare-band checkpoint out of the recommended release.
- Release bundles contain inference weights, portable metadata, model card, MIT license and SHA-256 checksums; exclude optimizer state, corpora and machine-specific/private metadata.
- Preparing an archive does not publish it. Commit/push/tag/release actions should follow the user's requested scope. Do not change repository visibility implicitly.
- Add a pretrained download link only after the actual release asset is available and verified. Do not assume a tag, release or URL exists. See `docs/PUBLISHING.md`.

## Commits and Pull Requests

Use short, imperative commit summaries. Explain the final behavior, relevant validation and material limitations. Include reproducible commands and plots when changing generated data or model behavior. Preserve unrelated existing changes, and never include large local artifacts merely because they are present in the workspace.
