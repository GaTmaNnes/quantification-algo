# Makefile

.PHONY: setup check demo

setup:
	./scripts/setup.sh

check:
	python3 scripts/check_environment.py

demo:
	python3 scripts/run_demo.py
