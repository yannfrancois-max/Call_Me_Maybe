.PHONY: install run debug clean fclean re lint

export UV_CACHE_DIR ?= /goinfre/$(USER)/.cache/uv
GOINFRE_VENV := /goinfre/$(USER)/Call_Me_Maybe_venv

install:
		mkdir -p $(GOINFRE_VENV)
		@if [ ! -L .venv ] || [ "$$(readlink .venv)" != "$(GOINFRE_VENV)" ]; then \
			rm -rf .venv; \
			ln -s $(GOINFRE_VENV) .venv; \
		fi
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

re: fclean install

fclean: clean
		rm -rf .venv
		rm -rf $(GOINFRE_VENV)
		uv cache clean

lint:
		uv run flake8 src
		uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
		uv run flake8 src
		uv run mypy --strict src