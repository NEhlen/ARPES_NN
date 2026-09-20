# Model card: compact ARPES denoiser

## Intended use

Exploratory denoising of nonnegative energy–momentum/angle intensity images and evaluation of synthetic-data training. The model is a small residual U-Net, not a physical inverse solver. It is not intended to determine a self-energy, establish a small band gap, or replace raw-data analysis.

The code is distributed separately from trained weights. The local reference checkpoint is `training_v2_20260919/denoise/best_model.pt`; there is no assumed public download. The prepared release bundle includes the MIT license and can be uploaded separately. Synthetic figures and compact benchmark summaries are included so results remain readable without the checkpoint.

## Inputs and outputs

- One intensity channel; reference training grid 256×256.
- Energy along rows; momentum/angle along columns.
- Input divided by its mean; predicted correction added to input and clipped nonnegative.
- No coordinates, material identifiers, count estimate or uncertainty output.
- The `denoise.py` CLI restores input intensity units and saves results on the model grid. It does not restore spatial/spectral resolution lost in resampling.
- Training covers positive measurements, not negative background-subtracted arrays.

## Training and selection

10,000 synthetic configurations from ten band families, with 1,000 independent validation configurations. Fresh Poisson noise and horizontal flips during training. MSE against the same measurement before counting noise; self-energy, occupation, resolution, envelope and background remain in the target. Fifty epochs, batch 32, AdamW, validation-plateau scheduling and best-validation checkpoint selection. See [training details](TRAINING.md).

Reference environment: Python 3.12, PyTorch 2.10.0+cu128, NVIDIA RTX 4060 Ti 16 GB. Approximately 79 minutes for the historical **two-model** run. No runtime claim is made for a different GPU or denoising-only run.

## Evaluation

1,000 independent synthetic test configurations plus 1,000 configurations from Mexican-hat/saddle families excluded from training. A separate overlapping-component benchmark uses 648 test images and independent filter/detection calibration. See [benchmark report](BENCHMARKS.md) and [machine-readable results](results/).

The network improves tested synthetic MSE and many component-fitting outcomes. Severe overlap at low counts produces failures. Naive post-processing fit error bars under-cover truth. The experimental figure has no clean reference, so it supports no numerical accuracy claim.

## Limitations and failure modes

- Shared simulator assumptions limit the strength of held-out-family results.
- Poisson noise and simple smooth backgrounds are not a complete instrument model.
- Finite resolution, pixel scale and resampling change feature statistics.
- Weak features can be attenuated, shifted or altered; inspect residuals.
- A visibly smooth image can still lead to a biased or ill-conditioned fit.
- Two fitted components do not establish that two identifiable bands exist.
- Scalar MSE can hide scientifically important local errors.
- This model supplies no calibrated uncertainties. Errors in denoised pixels are correlated.

Evaluate against independent repeats/high-count references and compare raw-data fits before using derived quantities. Improvements on controlled synthetic Voigt profiles are not evidence of reliable recovery of arbitrary interacting or waterfall spectra.

## Provenance and distribution

The exact reference checkpoint hash is recorded in `docs/results/denoising.json`. Benchmark summaries contain no local absolute paths. Raw experimental acquisitions, labbooks, full training arrays and optimizer states are excluded from Git. A release archive should contain only inference weights, portable metadata, this model card and checksums—not private measurements or a local environment dump.

## License

The project code, documentation and prepared pretrained-denoiser weights are distributed under the [MIT license](../LICENSE).
