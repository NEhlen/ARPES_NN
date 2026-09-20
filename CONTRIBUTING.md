# Contributing

This is an exploratory ARPES denoising project. Small, reproducible improvements are welcome, especially realistic noise models, experimental repeat-acquisition validation and tests of fitted-parameter bias.

## Development

Use Python 3.12, run `uv sync --locked --extra dev`, then:

```bash
uv run python -m unittest discover -s tests -v
uv run black arpesnn tests
```

Tests generate their own small fixtures. The optional local SP2 test is skipped when private measurements are absent. CPU-only testing is supported. The GitHub workflow uses a CPU PyTorch wheel; reference training used CUDA.

## Changes to simulation or analysis

State which physical assumptions change, keep random seeds reproducible, and add a small regression test for a meaningful invariant. For preprocessing changes, show shared-scale plots and inspect line shapes/residuals. Report parameter bias or failure cases as well as image metrics. Do not tune a method on the final test set or interpret independent-pixel covariance as calibrated uncertainty after denoising.

Do not commit generated corpora, checkpoints, raw acquisitions, labbook scans, access tokens or local environment files. Curated figures and compact summaries belong under `docs/`, with provenance. Make sure you have permission to share any experimental data or derived figure.

The original bare-band workflow is retained for compatibility; focus new user-facing work on denoising unless a change explicitly concerns that separate exploratory task.

This project uses the [MIT license](LICENSE). Contributions are accepted under the same license.
