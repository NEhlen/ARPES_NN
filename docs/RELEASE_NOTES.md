# v0.2.0 — exploratory background estimation

Release candidate; prepared locally for publication as a **prerelease**.

Adds an optional additive-background estimator alongside the existing ARPES denoiser. It retains the estimated background and signed corrected spectrum in input intensity units. Subtraction does not remove background shot noise, determine a bare band or identify which experimental intensity is intrinsic versus extrinsic.

## Included

- Background simulation with zero-background controls and broad/weak signal features; both original v1 and balanced v2 sampling profiles are reproducible.
- Background-v2 training with equal coverage of seven strength/contrast regimes, including locally buried bands, and a fixed total expected count budget.
- Separate compact U-Net, resumable GPU/CPU training, corpus audit and corrected-SP2/NumPy inference.
- Paired v1/v2 evaluation on identical held-out images, metrics around bands, repeated component fits, and five experimental previews.
- Explicit background colorbars, with shared truth/estimate scales and signed numerical subtraction.
- CPU regression tests and small denoising/background training/inference checks in CI.

## Results and limitations

On harder synthetic test data, mean relative ridge subtraction error decreases from 0.431 (v1) to 0.106 (v2). Under locally comparable backgrounds, close-doublet center error improves from 59.6 to 3.6 meV and weak-doublet error from 29.0 to 4.9 meV. These are controlled synthetic tests with known component counts, matching fitting models and twelve noise repeats per condition.

The buried close doublet still has roughly 100 meV position error; some easier cases regress. Band-shaped subtraction artifacts remain. Five experimental previews subtract about 13–43% of integrated intensity, but there is no decomposition ground truth. Neither model should be described as reliably preserving all bands.

The denoiser and its weights are unchanged. Signed background-subtracted arrays are outside its training distribution; combining the models has not been validated. The older bare-band workflow remains exploratory.

## Release attachments

- `arpes-denoiser-v1.tar.gz`: unchanged reference denoiser weights, with refreshed documentation.
- `arpes-background-v2.tar.gz`: epoch-36 background-v2 inference weights and task-specific model card.
- `SHA256SUMS.txt`: checksums for the two archives; each archive also contains internal file checksums.

Archives include portable model metadata and the MIT license. They exclude optimizer state, raw measurements, labbooks and generated training data. V1 background results remain available in the source documentation, but its weights are not the primary asset in this release.

See the [background guide](BACKGROUND.md), [figure provenance](FIGURES.md) and [publication instructions](PUBLISHING.md).
