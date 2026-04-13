.PHONY: help data curves factors signals backtest evaluate report website-build clean run figures \
       run-from-scratch all test lint lint-fix format typecheck coverage check catalog validate \
       env install clean-all

PKG = commodity_curve_factors
PYTHON = python
CONDA_ENV = curve-factors

## —— Help ——————————————————————————————————————————————
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## —— Pipeline ——————————————————————————————————————————
data: ## Download all market data (futures, inventory, macro, CFTC)
	$(PYTHON) -m $(PKG).data

curves: ## Build daily term structures for all commodities
	$(PYTHON) -m $(PKG).curves

factors: ## Compute all factor signals (carry, slope, curvature, momentum, inventory)
	$(PYTHON) -m $(PKG).factors

signals: ## Generate portfolio signals and weights
	$(PYTHON) -m $(PKG).signals.portfolio

backtest: ## Run backtests for all strategy variants
	$(PYTHON) -m $(PKG).backtest

evaluate: ## Compute performance metrics and attribution
	$(PYTHON) -m $(PKG).evaluation.report

report: ## Generate visualization tearsheet
	$(PYTHON) -m $(PKG).visualization.tearsheet

all: data curves factors signals backtest evaluate report ## Run full pipeline end-to-end

## —— Convenience ———————————————————————————————————————
run: ## Run pipeline via script
	$(PYTHON) scripts/run_pipeline.py

figures: ## Generate all publication-quality charts
	$(PYTHON) scripts/generate_figures.py

run-from-scratch: data run figures website-build ## Full pipeline from raw data to website
	@echo "Full pipeline complete. Results in results/, website in website/"

catalog: ## Print data catalog (row counts, date ranges, file sizes)
	$(PYTHON) -m $(PKG).data.catalog

validate: ## Run data validation spot-checks
	$(PYTHON) -m $(PKG).data.validate

## —— Quality ——————————————————————————————————————————
lint: ## Run ruff linter
	ruff check src/ tests/

lint-fix: ## Auto-fix ruff violations
	ruff check --fix src/ tests/

format: ## Auto-format code with ruff
	ruff format src/ tests/ scripts/

format-check: ## Check formatting without modifying files
	ruff format --check src/ tests/ scripts/

typecheck: ## Run mypy type checker
	mypy src/$(PKG)/ --ignore-missing-imports

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src/$(PKG) --cov-report=term-missing --cov-report=html:htmlcov

check: lint format-check typecheck test ## Run all quality checks (lint + format + types + tests)

## —— Website ——————————————————————————————————————————
website-build: ## Copy figures to website assets
	mkdir -p website/assets/figures
	cp results/figures/*.png website/assets/figures/ 2>/dev/null || true

## —— Setup ————————————————————————————————————————————
env: ## Create conda environment
	conda create -n $(CONDA_ENV) python=3.12 -y
	@echo "Run: conda activate $(CONDA_ENV) && make install"

install: ## Install package in editable mode with dev deps
	pip install -e ".[dev]"

## —— Cleanup ——————————————————————————————————————————
clean: ## Remove processed data and results
	rm -rf data/processed/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf results/tearsheets/*

clean-all: clean ## Remove everything including raw data and caches
	rm -rf data/raw/futures/*
	rm -rf data/raw/inventory/*
	rm -rf data/raw/macro/*
	rm -rf data/raw/cftc/*
	rm -rf data/cache/*
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
