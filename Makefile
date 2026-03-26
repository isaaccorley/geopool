.PHONY: install check test

install:
	uv sync --all-groups

check:
	uv run ruff format
	uv run ruff check --fix --unsafe-fixes
	uv run ty check

test:
	uv run pytest -vvv
