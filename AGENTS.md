# AGENTS.md

Repo: geopool. Python 3.13. Package manager: uv.

## Quick Start
- Install: `uv sync --dev`
- Run module scripts via `uv run python <path>`

## Build / Lint / Test
- Format: `uv run ruff format`
- Lint (fix): `uv run ruff check --fix --unsafe-fixes`
- Type check: `uv run ty check`
- Tests (full): `uv run pytest -v`
- Tests (default opts from pyproject): `uv run pytest`
- Tests (single file): `uv run pytest tests/test_file.py`
- Tests (single test): `uv run pytest tests/test_file.py::test_name`
- Tests (by keyword): `uv run pytest -k "keyword"`
- Tests (no xdist): `uv run pytest -n 0`

Notes:
- pytest addopts: `-v --tb=short -n auto` (see pyproject)
- For fast local loops: run format + lint + typecheck before pytest.

## Execution Examples
- kNN probe: `uv run python scripts/knnprobe.py --dataset-name aef`
- Linear probe: `uv run python scripts/linearprobe.py --dataset-name aef`
- Tables: `uv run python scripts/knn_table.py --output paper/knn_table.tex`
- Plot: `uv run python scripts/plot_results.py`

## Repo Conventions
- Source: `src/geopool/`
- Tests: `tests/`
- Data utilities: `data/`, `scripts/`
- Inputs often numpy arrays with shapes `(H, W, D)` or `(N, H, W, D)`

## Imports
- Standard library first, third-party next, local last.
- Use explicit imports; avoid `import *`.
- `from __future__ import annotations` used in modules that need forward refs.
- `TYPE_CHECKING` guard for type-only imports.

## Formatting
- Ruff governs formatting: line length 100.
- Use trailing commas where Black-style formatting expects it.
- Prefer double quotes for strings (ruff default).
- Keep files compact; split if >~500 LOC.

## Types
- Type hints required for public functions and methods.
- Use `np.ndarray` for arrays, with shape described in docstring if needed.
- Prefer `list[T]`, `dict[K, V]`, `tuple[...]` and `|` unions.
- Use `Literal[...]` for constrained string args.
- Avoid `Any` unless unavoidable; `ANN401` ignored but still prefer concrete types.

## Naming
- Functions/vars: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Pool methods prefixed with `pool_`.

## Docstrings
- Short, imperative summary line.
- Document shapes when non-obvious.
- Numpydoc validation configured; keep sections consistent when adding detailed docs.

## Error Handling
- Validate inputs early; raise `ValueError` for bad shapes or params.
- Use `RuntimeError` for invalid call order (e.g., fit before transform).
- Prefer explicit error messages with context (values, shapes, dims).
- Avoid silent failures; sanitize and warn when replacing invalid data.

## Performance
- Favor vectorized numpy over Python loops.
- If looping needed, keep loops tight; preallocate arrays.
- Use `astype(..., copy=False)` to avoid unnecessary copies.
- Be mindful of memory when reshaping large embeddings.

## Testing
- Tests use pytest + xdist. Keep tests deterministic.
- Use small synthetic arrays for unit tests; avoid large downloads.
- If you fix a bug, add/extend a regression test.

## Logging / Output
- Avoid heavy logging; prefer clear warnings (`print`) when needed.
- For user-facing scripts, keep CLI output concise.

## Data / IO
- Paths via `pathlib.Path`.
- Ensure GeoTIFF read/write uses rasterio conventions.
- Keep IO isolated from pure compute helpers.

## Design Patterns
- Functions in `pool.py` operate on `(H, W, D)` and `(N, H, W, D)`.
- Use helpers like `_spatial_axes` to centralize shape logic.
- Keep pool methods pure and deterministic.

## Dependency Notes
- Third-party: numpy, sklearn, rasterio, torch, torchgeo, xarray.
- For optional heavy deps, guard imports if adding new modules.

## Lint Rules (Ruff)
- Extended selections include annotations, bugbear, comprehensions, pytest style.
- Per-test ignores exist under `tests/**`.
- Line length 100, target-version py312.

## Agent Expectations
- Run format + lint + typecheck before tests when changing code.
- Prefer minimal diffs; keep style consistent with existing modules.
- Update docs when behavior changes (README or module docstrings).
