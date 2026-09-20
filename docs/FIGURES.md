# Figure provenance

These are numerical results or experimental previews, not generated artwork. The reference denoiser is the best-validation checkpoint from the 50-epoch `training_v2_20260919` run. Its hash is in `results/denoising.json`.

| Figure | Source and selection |
|---|---|
| `synthetic-denoising.png` | Four lowest-count test examples with distinct families; `plot_low_counts.py`. Source test indices/counts are recorded below. Shared intensity scales within rows. |
| `experimental-denoising.png` | Corrected SP2 block from local acquisition `mos2_2_074.sp2`; `plot_trained_examples.py`. Original, resized input, prediction and amplified residual. First three panels share a scale. No reference ground truth. |
| `error-by-noise.png` | Full test and unseen-family errors, `benchmark_denoising.py`. Five count bins; filter settings selected on validation data. |
| `overlapping-bands.png` | Full repeated-noise component-position errors, `report_overlap.py`; nine configurations, three count levels, 24 repeats. |
| `edc-fits.png` | First noise realization at 30 counts for three prespecified overlapping cases, central EDC; `report_overlap.py`. Illustrative, not selected for best performance. |

The experimental image is a qualitative preview. Raw measurement files and the labbook are not included. No material identification, sample provenance or clean-spectrum claim is inferred from the filename. Reproducing that exact figure requires the original private acquisition; reproducing synthetic benchmarks does not.

The historical plotting helpers retain their original local-run defaults. Public benchmark CLIs accept explicit paths; their outputs can regenerate the corresponding benchmark figures for another checkpoint.

Synthetic test selection:

```json
[
  {
    "test_index": 920,
    "family": "parabolic",
    "counts_per_mean_pixel": 3.3034675121307373,
    "global_snr_dB": 14.839138984680176
  },
  {
    "test_index": 87,
    "family": "avoided_crossing",
    "counts_per_mean_pixel": 3.319007158279419,
    "global_snr_dB": 14.320520401000977
  },
  {
    "test_index": 435,
    "family": "square_lattice",
    "counts_per_mean_pixel": 3.3458058834075928,
    "global_snr_dB": 8.887144088745117
  },
  {
    "test_index": 158,
    "family": "triangular_lattice",
    "counts_per_mean_pixel": 3.3794047832489014,
    "global_snr_dB": 15.3881254196167
  }
]
```
