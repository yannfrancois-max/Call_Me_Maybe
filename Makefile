.PHONY: install run debug clean lint

export UV_CACHE_DIR ?= /goinfre/$(USER)/.cache/uv

install:
		uv sync

run:
		uv run  --active python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json $(ARGS)

debug:
		uv run python -m pdb -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json $(ARGS)

clean:
		find . -type d -name "__pycache__" -exec rm -rf {} +
		find . -type d -name ".mypy_cache" -exec rm -rf {} +
		find . -type d -name ".pytest_cache" -exec rm -rf {} +
		rm -rf data/output/*

lint:
		uv run flake8 src
		uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
		uv run flake8 src
		uv run mypy --strict src