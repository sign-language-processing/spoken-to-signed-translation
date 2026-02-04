.PHONY: check format test

# Check linting and formatting issues
check:
	ruff check spoken_to_signed
	ruff format --check spoken_to_signed

# Format source code automatically
format:
	ruff check --fix spoken_to_signed
	ruff format spoken_to_signed

# Run tests for the package
test:
	python -m pytest
