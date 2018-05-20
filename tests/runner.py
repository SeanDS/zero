import sys
from unittest import TestLoader, TextTestRunner

loader = TestLoader()

def find_tests(suite_name):
    return loader.discover('tests.' + suite_name)

def run_suite(suite):
    runner = TextTestRunner(verbosity=3)
    runner.run(suite)

# test suites
suites = {
    "unit": find_tests('unit'),
    "integration": find_tests('integration'),
    "validation": find_tests('validation')
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter the name of a test suite to run, or \"all\" to run all:")
        
        for test in suites.keys():
            print("\t%s" % test)
    else:
        suite_name = sys.argv[1]

        if suite_name == "all":
            print("Running all test suites")
            for suite in suites.values():
                run_suite(suite)
        else:
            print("Running %s test suite" % suite_name)
            run_suite(suites[suite_name])