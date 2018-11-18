"""LISO vs native solver tests"""

import sys
import os.path
import glob
import unittest
import logging

from zero import set_log_verbosity, add_log_handler
from zero.liso import LisoRunner

# directory containing tests relative to this script
REL_FIL_DIR = "scripts"


class LisoTester(unittest.TestCase):
    def __init__(self, method_name, fil_path=None):
        super().__init__(method_name)

        self.fil_path = fil_path

    def test_liso_vs_native(self):
        # test message
        message = "Test %s against LISO" % self.fil_path

        with self.subTest(msg=message + " (without prescaling)"):
            self.compare(self._liso_output(), prescale=False)

        with self.subTest(msg=message + " (with prescaling)"):
            self.compare(self._liso_output(), prescale=True)

    def compare(self, liso_output, **native_kwargs):
        # get LISO solution
        liso_solution = liso_output.solution()

        # run native
        native_solution = liso_output.solution(force=True, **native_kwargs)

        # check if they match (only check defaults as LISO only generates defaults)
        self.assertTrue(liso_solution.equivalent_to(native_solution, defaults_only=True))

    def _liso_output(self):
        # run LISO and parse output
        return LisoRunner(self.fil_path).run()

    @property
    def description(self):
        """LISO file top comment, if any"""

        text = ""

        with open(self.fil_path, "r") as obj:
            next_line = obj.readline()

            while next_line.startswith("#"):
                text += next_line.lstrip("#")

                # read next line
                next_line = obj.readline()

        if text == "":
            text = "no description"
        else:
            # remove extra whitespace, newlines, etc. if present
            text = "\"" + text.strip() + "\""

        return text

def fil_scripts():
    # find *.fil scripts in all subdirectories
    return glob.glob(os.path.join(fil_dir(), "**/*.fil"), recursive=True)

def fil_dir():
    this_script_dir = os.path.dirname(os.path.realpath(__file__))

    return os.path.abspath(os.path.join(this_script_dir, REL_FIL_DIR))

def load_tests(loader, tests, pattern):
    test_cases = unittest.TestSuite()

    for script in fil_scripts():
        test_cases.addTest(LisoTester("test_liso_vs_native", script))

    return test_cases

if __name__ == '__main__':
    # run individual test
    if len(sys.argv) < 2:
        print("Enter a LISO test path")
        sys.exit(1)

    if len(sys.argv) > 2:
        VERBOSITY = int(sys.argv[2])

        if VERBOSITY < 0:
            raise ValueError("verbosity must be > 0")
        elif VERBOSITY > 2:
            VERBOSITY = 2

        # tune in to zero's logs
        LOGGER = logging.getLogger("zero")
        # show only warnings with no verbosity, or more if higher
        set_log_verbosity(logging.WARNING - 10 * VERBOSITY, LOGGER)
    else:
        VERBOSITY = 0

    PATH = sys.argv[1]

    SUITE = unittest.TestSuite()
    SUITE.addTest(LisoTester("test_liso_vs_native", PATH))

    RUNNER = unittest.TextTestRunner(verbosity=VERBOSITY, stream=sys.stdout)
    sys.exit(not RUNNER.run(SUITE).wasSuccessful())
