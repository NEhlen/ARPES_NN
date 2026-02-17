# Repository Guidelines

## Project Structure & Module Organization
- `arpesnn/` holds the core Python modules for ARPES spectrum generation and the PyTorch U-Net model.
- `arpesnn/dataset/` is a placeholder for generated training data (kept with `.gitkeep`).
- `arpesnn/models/` stores trained model checkpoints (also tracked with `.gitkeep`).
- `pyproject.toml` defines project metadata and dependencies managed by `uv`.

## Build, Test, and Development Commands
- `uv venv` creates the local virtual environment in `.venv`.
- `uv sync` installs runtime dependencies from `pyproject.toml`.
- `uv sync --extra dev` installs runtime + dev dependencies (includes `black`).
- `python arpesnn/generate_data.py` generates training data; requires `DATASET_PATH` in your `.env`.
- `python arpesnn/nn_pytorch.py` trains the U-Net on the dataset and saves a model.
- `black arpesnn` formats the codebase.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard PEP 8 style.
- Module names are lowercase with underscores (e.g., `spectral_function.py`).
- Classes use `CamelCase`; functions and variables use `snake_case`.
- Format with `black` before committing.

## Testing Guidelines
- No automated tests are currently present.
- If you add tests, prefer `pytest` and keep tests in `tests/` with names like `test_<module>.py`.

## Commit & Pull Request Guidelines
- The repository has no commits yet, so no commit message convention is established.
- Suggested convention: short, imperative summaries (e.g., `Add spectrum generator CLI`).
- PRs should include a clear description of changes, reproduction steps, and plots or sample outputs when data generation or model behavior changes.

## Configuration Tips
- Create a `.env` file with `DATASET_PATH` pointing to a writable dataset directory (example: `/home/niels/projects/ARPES_NN/arpesnn/dataset/`).
- Set `MODELS_PATH` to where trained checkpoints should be written (example: `/home/niels/projects/ARPES_NN/arpesnn/models/`).
- Keep generated data and model checkpoints out of version control unless explicitly requested.
