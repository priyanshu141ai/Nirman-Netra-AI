.PHONY: install format lint type-check test run-api run-dashboard compose-validate

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

type-check:
	uv run mypy

test:
	uv run pytest

run-api:
	uv run uvicorn nirman_netra.api.main:app --reload

run-dashboard:
	uv run streamlit run src/nirman_netra/dashboard/app.py

compose-validate:
	docker compose config --quiet
