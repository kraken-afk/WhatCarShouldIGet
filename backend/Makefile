LINT_TARGET=.
MANAGE=./src/manage.py

lint:
	ruff check --fix $(LINT_TARGET)

format:
	ruff format $(LINT_TARGET)

dev:
	$(MANAGE) runserver

