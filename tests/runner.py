"""Test suite runner"""

import sys
import os.path
import logging
from unittest import TestLoader, TextTestRunner

from circuit import set_log_verbosity

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

def run_and_exit(suite, verbosity=1):
    """Run tests and exit with a status code representing the test result"""
    runner = TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())

def find_tests(suite_path):
    """Find tests at the specified location"""
    return LOADER.discover(suite_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter the name of a test suite to run, or \"all\" to run all:")

        for test in SUITES:
            print("\t%s" % test)

        sys.exit(1)

    SUITE_NAME = sys.argv[1]

    if len(sys.argv) > 2:
        VERBOSITY = int(sys.argv[2])

        if VERBOSITY < 0:
            raise ValueError("verbosity must be > 0")
        elif VERBOSITY > 2:
            VERBOSITY = 2

        # tune in to circuit's logs
        LOGGER = logging.getLogger("circuit")
        # show only warnings with no verbosity, or more if higher
        set_log_verbosity(logging.WARNING - 10 * VERBOSITY, LOGGER)
    else:
        VERBOSITY = 0

    if SUITE_NAME == "all":
        print("Running all test suites")
        run_and_exit(find_tests(os.path.join(THIS_DIR, '.')), verbosity=VERBOSITY)
    else:
        print("Running %s test suite" % SUITE_NAME)
        run_and_exit(find_tests(os.path.join(THIS_DIR, SUITES[SUITE_NAME])), verbosity=VERBOSITY)
