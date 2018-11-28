docs-html:
	$(MAKE) -C docs html

test: test-all

test-all:
	python -m tests run all

test-unit:
	python -m tests run unit

test-integration:
	python -m tests run integration

test-validation:
	python -m tests run validation

lint:
	pylint --rcfile=.pylintrc zero

security:
	bandit -c=.banditrc -r zero
