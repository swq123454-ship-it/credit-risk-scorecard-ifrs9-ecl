.PHONY: help setup pipeline pipeline-real test lint clean tree

help:
	@echo "Credit risk scorecard + IFRS 9 ECL engine"
	@echo ""
	@echo "  make setup         install dependencies"
	@echo "  make pipeline      run the full pipeline on the synthetic book"
	@echo "  make pipeline-real run on a LendingClub extract placed in data/raw/loan.csv"
	@echo "  make test          run the test suite"
	@echo "  make lint          run ruff"
	@echo "  make clean         remove generated data, figures and tables"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

pipeline:
	python scripts/run_pipeline.py

pipeline-real:
	python scripts/run_pipeline.py --data-mode real

test:
	pytest

lint:
	ruff check .

clean:
	rm -f data/interim/*.csv
	rm -f data/processed/*.db
	rm -f outputs/figures/*.png
	rm -f outputs/tables/*.csv
	rm -f outputs/metrics.json outputs/RESULTS.md
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
