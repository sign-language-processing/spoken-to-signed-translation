.PHONY: check format test

# Check formatting issues
check:
	pylint --rcfile=.pylintrc src
	yapf -dr src

# Format source code automatically
format:
	isort --profile black src
	yapf -ir src

# Run tests for the package
test:
	python -m pytest
