"""LISO vs native solver test suite"""

import os
import glob
from unittest import TestSuite, TestCase

from zero.liso import LisoRunner


class LisoTestSuite(TestSuite):
    def __init__(self, search_path):
        super().__init__()

        self._find_tests(search_path)

    def _find_tests(self, search_path):
        # find .fil scripts
        for script in glob.glob(os.path.join(search_path, "**/*.fil"), recursive=True):
            self.addTest(LisoComparisonTest(script))


class LisoComparisonTest(TestCase):
    def __init__(self, path):
        super().__init__()

        self.script_path = path

    def runTest(self):
        # test message
        message = f"Test {self.script_path} against LISO"

        with self.subTest(msg=message):
            self._compare()

    def _compare(self):
        # get parsed LISO output
        liso_output = self._liso_result()

        # get LISO solution
        liso_solution = liso_output.solution()

        # run native
        native_solution = liso_output.solution(force=True)

        # check if they match (only check defaults as LISO only generates defaults)
        self.assertTrue(liso_solution.equivalent_to(native_solution, defaults_only=True))

    def _liso_result(self):
        # run LISO and parse output
        return LisoRunner(self.script_path).run()

    @property
    def description(self):
        """LISO file top comment, if any"""
        text = ""

        with open(self.script_path, "r") as obj:
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
