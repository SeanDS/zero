import sys
import os.path
from unittest import TestLoader, TextTestRunner

# this directory
this_dir = os.path.dirname(os.path.realpath(__file__))

loader = TestLoader()

def find_tests(suite_name):
    return loader.discover(suite_name)

def run_suite(suite):
    runner = TextTestRunner(verbosity=3)
    runner.run(suite)

# test suites
suites = {
    # suite name / directory
    "unit": "unit",
    "integration": "integration",
    "validation": "validation"
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter the name of a test suite to run, or \"all\" to run all:")
        
        for test in suites:
            print("\t%s" % test)
    else:
        suite_name = sys.argv[1]

        if suite_name == "all":
            print("Running all test suites")
            run_suite(find_tests(os.path.join(this_dir, '.')))
        else:
            print("Running %s test suite" % suite_name)
            run_suite(find_tests(os.path.join(this_dir, suites[suite_name])))