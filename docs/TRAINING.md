# Simulation and training

For the separate background task, see [background generation, training and reference results](BACKGROUND.md). Its targets and checkpoint are distinct from denoising.

## Reproduce the reference-sized corpus

From the repository root, with dependencies installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python arpesnn/corpus.py \
  --output arpesnn/dataset/denoising-v1 \
  --train 10000 --validation 1000 --test 1000 --ood 1000 --size 256 --workers 8
uv run python arpesnn/audit_corpus.py arpesnn/dataset/denoising-v1
uv run python arpesnn/train_tasks.py \
  --corpus arpesnn/dataset/denoising-v1 --output arpesnn/models/denoising-v1 \
  --task denoise --epochs 50 --batch-size 32 --workers 4 --require-cuda
```

Use fewer workers if RAM or CPU resources are limited. Omit `--require-cuda` to allow CPU training. A compatible GPU-enabled PyTorch installation and NVIDIA driver are needed for CUDA; check with `uv run python -c "import torch; print(torch.cuda.is_available())"`. The locked environment records the reference dependency versions but hardware/backend differences can affect results.

Generation refuses an existing corpus directory. Training requires a new output directory unless `--resume` is used. A full corpus occupies about 7 GB and took roughly six minutes to generate with eight workers on the original host. That timing is illustrative, not a requirement or a guarantee.

The historical reference run used `--task both`: separate denoising and bare-band models with independent weights/optimizers, sharing data loading. Denoising-only training above is the recommended workflow but need not reproduce identical numerical weights because RNG consumption differs. Both use the same architecture and denoising objective.

## Physics and targets

For each band, the simulator evaluates

```text
A(k, E) = [−Im Σ(E)] / {π [(E − ε(k) − Re Σ(E))² + (Im Σ(E))²]}.
```

A causal sum of retarded poles provides paired real and imaginary self-energy parts, with constant negative imaginary broadening and a real subtraction setting Re Σ(0)=0. This is a phenomenological analytic model, not a fitted material self-energy.

The families are parabolic, multiband parabolic, Dirac, Bernal-bilayer-like, Mexican hat, spin-split valence, square lattice, honeycomb, avoided crossing, saddle, triangular lattice and Rashba-like. Mexican hat and saddle occur only in the unseen-family (`ood`) split.

Cut center, direction, perpendicular offset and energy window vary. One branch is anchored inside the occupied window to avoid a predominantly empty dataset. Spectra include Fermi occupation, smooth intensity envelopes, background and Gaussian instrumental resolution. Four-by-four subpixel evaluation and bin averaging reduce discretization artifacts. Configurations with too little occupied-band coverage or excessively concentrated bare targets are resampled, with retry metadata retained. These acceptance rules also restrict the denoising distribution because both targets share a corpus.

The **denoising target** is the noise-free measurement after these effects. Physical broadening and renormalization are preserved. The second corpus channel is a legacy experimental bare-band target and is unused by `--task denoise`.

Counts per mean-intensity pixel are sampled log-uniformly between approximately 3.16 and 1,995. Local band peaks can therefore have much higher SNR than the average pixel. Noise is Poisson only; gain variations, correlated detector noise, dead pixels and background subtraction are not modeled comprehensively.

## Split integrity

Each split uses distinct underlying configurations and seed ranges. Training draws fresh noise each epoch; validation/test draws are fixed. Horizontal reflection is applied only during training. Mean normalization uses the noisy input, and the clean target receives the same scale. There is no experimental training data.

The audit checks finite/nonnegative images, mean normalization, seed/family separation, parameter hashes and exact regeneration of selected examples. The manifest records source hashes. Arrays are memory mapped; the corpus is not loaded into RAM wholesale.

## Architecture and optimization

`CompactUNet` uses two pooling stages, widths 16/32/64, two 3×3 convolution–GroupNorm–SiLU layers per block, bilinear upsampling and concatenated skip connections. Its 1×1 correction head is initialized to zero. Output is `max(input + correction, 0)`.

Training uses MSE, AdamW (learning rate 0.0003, weight decay 0.0001), gradient-norm clipping at 1, a ReduceLROnPlateau scheduler (factor 0.5, patience 3) and early stopping (patience 8). CUDA runs use mixed precision, channels-last tensors and pinned/prefetched batches. Validation selects the best checkpoint; the initial identity model is a candidate too.

## Outputs and resumption

- `denoise/best_model.pt`: compact inference bundle, selected by validation MSE.
- `denoise/best_model.metadata.json`: training settings and input shape.
- `last.pt`: optimizer/scheduler/scaler state and completed epoch for resumption.
- `metrics.csv`, `status.json`: progress and validation history.
- `source/`, `run_config.json`: source snapshot, dependency/hardware details and corpus hash.
- `evaluation.json`, `heldout_predictions.png`: final held-out evaluation and preview.

Resume using the same training command with `--resume`; only the last completed epoch is recoverable after interruption. Load only trusted resume checkpoints: `last.pt` contains more than inference weights. The public-facing inference CLI uses PyTorch's restricted `weights_only=True` loader.

## Legacy paths

`generate_tasks.py` is the earlier small dual-task pilot. `generate_data.py` and `nn_pytorch.py` implement the original bare-band workflow, and the tracked `example_dataset` belongs to that task. Do not relabel these targets as clean denoising references. `apply_model.py` retains older checkpoint/text-format support; its saved numeric predictions are normalized target units, unlike the new `denoise.py` CLI, which restores input intensity units.
