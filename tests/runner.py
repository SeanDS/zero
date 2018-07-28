"""Test suite runner"""

import sys
import os.path
from unittest import TestLoader, TextTestRunner

# this directory
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

# test suites
SUITES = {
    # suite name / directory
    "unit": "unit",
    "integration": "integration",
    "validation": "validation"
}

# test loader
LOADER = TestLoader()

def find_tests(suite_path):
    """Find tests at the specified location"""
    return LOADER.discover(suite_path)

def run_and_exit(suite, verbosity=1):
    """Run tests and exit with a status code representing the test result"""
    runner = TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter the name of a test suite to run, or \"all\" to run all:")

        for test in SUITES:
            print("\t%s" % test)
    else:
        SUITE_NAME = sys.argv[1]

        if len(sys.argv) > 2:
            VERBOSITY = int(sys.argv[2])
        else:
            VERBOSITY = 0

        if SUITE_NAME == "all":
            print("Running all test suites")
            run_and_exit(find_tests(os.path.join(THIS_DIR, '.')), verbosity=VERBOSITY)
        else:
            print("Running %s test suite" % SUITE_NAME)
            run_and_exit(find_tests(os.path.join(THIS_DIR, SUITES[SUITE_NAME])), verbosity=VERBOSITY)
