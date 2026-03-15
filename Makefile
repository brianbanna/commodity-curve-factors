.PHONY: data curves factors signals backtest evaluate report website clean all test

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

website:
	@echo "Copy key figures to website/assets/figures/ and open website/index.html"

clean:
	rm -rf data/processed/*
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf results/tearsheets/*

all: data curves factors signals backtest evaluate report

test:
	python -m pytest tests/ -v
