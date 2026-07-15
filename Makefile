.PHONY: install test lint format smoke audit-demo

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

smoke:
	python scripts/generate_demo_data.py --output data/processed/demo.csv
	python scripts/train.py --config configs/train_smoke.yaml
	python scripts/evaluate.py --config configs/evaluate.yaml
	python scripts/predict.py --input examples/demo_input.csv --output outputs/demo_predictions.csv

audit-demo:
	python scripts/audit_assets.py --input examples --output audit/demo

