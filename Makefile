docs-html:
	$(MAKE) -C docs html
	xdg-open docs/_build/html/index.html

test: test-all

test-unit:
	python -m tests run unit

test-integration:
	python -m tests run integration

test-validation:
	python -m tests run validation

test-validation-fast:
	python -m tests run validation-fast

test-all:
	python -m tests run all

test-all-fast:
	python -m tests run all-fast

lint:
	pylint --rcfile=.pylintrc zero

security:
	bandit -c=.banditrc -r zero
