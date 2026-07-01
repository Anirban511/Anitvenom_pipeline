# Antivenom Pipeline - common tasks
.PHONY: help install test run clean

help:
	@echo "make install   - install dependencies"
	@echo "make test      - run the unit test suite (25 tests)"
	@echo "make run       - run the full pipeline + analytics"
	@echo "make clean     - remove generated artifacts and caches"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python run_all.py

clean:
	rm -rf results data .cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
