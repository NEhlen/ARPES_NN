# ARPES_NN

Neural-network workflow for simulated ARPES spectra:
- generate synthetic training pairs (input spectrum with broadening/noise, target spectrum)
- train a U-Net model to reconstruct cleaner spectra

## Project Layout
- `arpesnn/generate_data.py`: data generation script
- `arpesnn/nn_pytorch.py`: model training script (CUDA-aware)
- `arpesnn/spectral_function.py`: spectral function and self-energy models
- `arpesnn/dispersion_relations.py`: dispersion abstraction
- `arpesnn/dataset/`: generated dataset folders (`001/`, `002/`, ...)
- `arpesnn/models/`: saved model checkpoints

## Requirements
- Python 3.12+
- `uv`
- Optional: NVIDIA GPU + CUDA for faster training

## Setup
```bash
uv venv
uv sync --extra dev
source .venv/bin/activate
```

## Configuration
Create `.env` in repo root:
```env
DATASET_PATH="/home/niels/projects/ARPES_NN/arpesnn/dataset/"
MODELS_PATH="/home/niels/projects/ARPES_NN/arpesnn/models/"
```

Notes:
- `DATASET_PATH` is used by `generate_data.py`
- `DATASET_PATH` and `MODELS_PATH` are used by `nn_pytorch.py`

## Generate Data
```bash
python3 arpesnn/generate_data.py
```

Behavior:
- The script checks existing numeric dataset folders in `DATASET_PATH`
- It finds the highest index and appends new data from the next index
- Default run generates 50 additional datasets

Each generated sample produces files like:
- `graphene_test_input.txt`
- `graphene_test_target.txt`
- `graphene_test_parameters.json`
- `graphene_test_image.png`

## Train Model
```bash
python3 arpesnn/nn_pytorch.py
```

Training highlights:
- automatic device selection (`cuda` if available)
- mixed precision on CUDA (AMP)
- per-epoch logs with loss, runtime, throughput, and peak GPU memory

Model output:
- saved to `${MODELS_PATH}/test_model`

## Development
Format code:
```bash
black arpesnn
```

Type-checking diagnostics can appear as annotations are tightened. Runtime behavior remains the same unless explicitly changed.

## Current Status
- No automated tests yet
- If adding tests, place them in `tests/` and run with `pytest`
