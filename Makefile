.PHONY: data curves factors signals backtest evaluate report website clean all test lint run figures run-from-scratch website-build

PKG = commodity_curve_factors

data:
	python -m $(PKG).data.futures_loader

curves:
	python -m $(PKG).curves.builder

factors:
	python -m $(PKG).factors.combination

signals:
	python -m $(PKG).signals.portfolio

backtest:
	python -m $(PKG).backtest.engine

evaluate:
	python -m $(PKG).evaluation.report

report:
	python -m $(PKG).visualization.tearsheet

website-build:
	mkdir -p website/assets/figures
	cp results/figures/*.png website/assets/figures/ 2>/dev/null || true

clean:
	rm -rf data/processed/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf results/tearsheets/*

run:
	python scripts/run_pipeline.py

figures:
	python scripts/generate_figures.py

run-from-scratch: data run figures website-build
	@echo "Full pipeline complete. Results in results/, website in website/"

all: data curves factors signals backtest evaluate report

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
