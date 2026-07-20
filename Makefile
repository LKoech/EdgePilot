.PHONY: setup setup-voice run run-voice serve test lint format typecheck ci eval clean

setup:
	python -m pip install -e ".[dev]" --quiet
	@echo ""
	@echo "EdgePilot installed. Ensure Ollama is running with a model pulled:"
	@echo "  ollama pull qwen2.5:3b"
	@echo ""

setup-voice:
	python -m pip install -e ".[dev,voice]" --quiet
	@echo ""
	@echo "EdgePilot installed with voice support (faster-whisper + piper)."
	@echo "Ensure Ollama is running: ollama pull qwen2.5:3b"
	@echo ""

run:
	python -m edgepilot.main

run-voice:
	python -m edgepilot.main --voice

serve:
	python -m edgepilot.main --serve

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/ eval/
	python -m ruff format --check src/ tests/ eval/

format:
	python -m ruff check --fix src/ tests/ eval/
	python -m ruff format src/ tests/ eval/

typecheck:
	python -m mypy src/edgepilot/

ci: lint test

eval:
	python -m eval.harness

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info
