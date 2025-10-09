.PHONY: .uv  # Check that uv is installed
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .ruff
.ruff:
	@ruff -V || echo 'Please install ruff: https://github.com/astral-sh/ruff?tab=readme-ov-file#getting-started'

.PHONY: .pre-commit # Check that pre-commit is installed
.pre-commit: .uv
	@uv run pre-commit -V || uv pip install pre-commit

.PHONY: install # Install all of the packages and their dependencies plus pre-commit for local development
install: .uv
	uv sync --frozen --group all --all-extras
	uv pip install pre-commit
	uv run pre-commit install --install-hooks

.PHONY: lock # Rebuild lockfiles from scratch, updating all dependencies
lock: .uv
	uv lock --upgrade

.PHONY: test
test:
	@uv run pytest -s

.PHONY: clean
clean:
	rm -rf `find .-name __pycache__`
	rm -rf .pytest_cache
	rm -rf site

.PHONY: docs # Generate documentation site
docs:
	uv run mkdocs build --strict

.PHONY: detect-secrets # Scan for and audit secrets, updating the .secrets.baseline
detect-secrets: .uv
	uv pip install --upgrade "git+https://github.com/ibm/detect-secrets.git@master#egg=detect-secrets"
	detect-secrets scan --update .secrets.baseline
	detect-secrets audit .secrets.baseline

.PHONY: lint
lint: .ruff
	uv run ruff check --fix

.PHONY: format
format: .ruff
	uv run ruff format
