.PHONY: check format test

# Check formatting issues
check:
	pylint --rcfile=.pylintrc spoken_to_signed
	yapf -dr spoken_to_signed

# Format source code automatically
format:
	isort --profile black spoken_to_signed
	yapf -ir spoken_to_signed

# Run tests for the package
test:
	python -m pytest
