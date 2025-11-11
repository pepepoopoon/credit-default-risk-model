PYTHON ?= python3.11

.PHONY: install lint test smoke train evaluate predict experiment

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest

smoke:
	PYTHONPATH=src $(PYTHON) -m credit_default_risk_model.data --output data/smoke.csv

train: smoke
	PYTHONPATH=src $(PYTHON) -m credit_default_risk_model.train --input data/smoke.csv --output-dir artifacts

evaluate:
	PYTHONPATH=src $(PYTHON) -m credit_default_risk_model.evaluate --input data/smoke.csv --artifact artifacts/model.joblib

predict:
	PYTHONPATH=src $(PYTHON) -m credit_default_risk_model.predict --input data/smoke.csv --artifact artifacts/model.joblib --output artifacts/scored.csv

experiment:
	PYTHONPATH=src $(PYTHON) -m credit_default_risk_model.experiment \
		--output experiments/results/001_baseline.json \
		--hypothesis "Зафиксировать baseline стоимости кредитных ошибок"
