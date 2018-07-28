test: test-all

test-all:
	python tests/runner.py all

test-unit:
	python tests/runner.py unit

test-integration:
	python tests/runner.py integration

test-validation:
	python tests/runner.py validation

lint:
	pylint --rcfile=.pylintrc circuit

security:
	bandit -c=.banditrc -r circuit
