.PHONY: install test lint format clean build docker

# Development
install:
	pip install -e ".[dev]"

test:
	pytest -v --cov=serendipity_finder --cov-report=term-missing

lint:
	ruff check .
	mypy serendipity_finder.py --ignore-missing-imports

format:
	black .
	ruff check --fix .

# Build
build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose up serendipity

docker-test:
	docker-compose --profile test up test

# Demo
demo:
	python serendipity_finder.py

# Help
help:
	@echo "Available targets:"
	@echo "  install     Install package with dev dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run linters"
	@echo "  format      Format code"
	@echo "  build       Build distribution package"
	@echo "  clean       Remove build artifacts"
	@echo "  docker-*    Docker commands"
	@echo "  demo        Run demonstration"
