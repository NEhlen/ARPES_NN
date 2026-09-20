# Publication notes

The release preparation is local. It does not commit, push, tag, upload or change repository visibility. The existing `v0.1.0` prerelease contains the original denoiser. The next candidate is **v0.2.0**, also a prerelease because background estimation remains exploratory.

## Source and private files

Commit code, tests, dependency lockfile, README, documentation, curated figures and compact numeric summaries. The tracked `example_dataset` remains a small historical bare-band example. Keep `.env`, virtual environments, corpora, checkpoints, raw `data/`, labbooks, `outputs/` and `releases/` local. Review both `git status` and `git ls-files`; ignore rules do not remove tracked files.

## Prepare the two release assets

From the repository root, with the trained checkpoints available locally:

```bash
uv run python scripts/prepare_release.py \
  --checkpoint arpesnn/models/training_v2_20260919/denoise/best_model.pt \
  --output releases/v0.2.0/arpes-denoiser-v1
uv run python scripts/prepare_release.py \
  --checkpoint arpesnn/models/background_v2_20260920/best_model.pt \
  --output releases/v0.2.0/arpes-background-v2
(cd releases/v0.2.0 && sha256sum arpes-denoiser-v1.tar.gz arpes-background-v2.tar.gz > SHA256SUMS.txt)
```

Choose a fresh destination if already prepared. Each archive includes inference weights, a task-specific model card, portable metadata, MIT license and internal checksums. `SHA256SUMS.txt` beside the archives verifies downloads. The denoiser weights are unchanged from v0.1.0; its archive documentation is refreshed. Model version names (`denoiser-v1`, `background-v2`) are separate from the repository release version (`v0.2.0`).

Do not upload optimizer/resume checkpoints (`last.pt`), raw acquisitions, corpora or labbooks. The original background-v1 results stay documented, but v2 is the background asset for this candidate. Both are exploratory.

## Verify from a clean copy

```bash
uv sync --locked --extra dev
uv run python -m unittest discover -s tests -v
uv run black --check arpesnn tests scripts
```

Run both CPU smoke workflows in `.github/workflows/tests.yml` without private data or local models. Check README links and figures, and use the packaged checkpoints for inference on a synthetic example. A local pass does not imply GitHub Actions has run.

After extraction, use `denoise.py --checkpoint /path/to/arpes-denoiser-v1/denoiser.pt` or `apply_background.py --checkpoint /path/to/arpes-background-v2/background.pt`, with `--input` and a new `--output` directory. The reference models use 256×256 grids. Background output is signed and must not be passed directly into the denoiser.

## Commit, push and publish

Review the candidate changes and commit them on the intended branch. Push that branch and let GitHub Actions finish. Then tag the **reviewed commit**, not a moving branch tip:

```bash
git tag -a v0.2.0 -m "ARPES NN v0.2.0: exploratory background estimation" <reviewed-commit>
git push origin v0.2.0
gh release create v0.2.0 --repo NEhlen/ARPES_NN --verify-tag --prerelease \
  --title "ARPES NN v0.2.0 — exploratory background estimation" \
  --notes-file releases/v0.2.0/RELEASE_NOTES.md \
  releases/v0.2.0/arpes-denoiser-v1.tar.gz \
  releases/v0.2.0/arpes-background-v2.tar.gz \
  releases/v0.2.0/SHA256SUMS.txt
```

The local release-notes copy should use absolute links to documentation at the reviewed commit or release tag. Remove the candidate-only status line before publication. Verify the public assets and their checksums after upload, then add the new download link to README. Do not assume the v0.2.0 URL exists before publishing it. The source release notes live in `docs/RELEASE_NOTES.md`.
