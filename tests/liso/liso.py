import os.path
import glob
import numpy as np
import unittest

from electronics import logging_on
logging_on()
from electronics.simulate.liso import Runner

class TestLisoVsNative(unittest.TestCase):
    REL_FIL_DIR = "."

    def test_scripts(self):
        for script in self.fil_scripts:
            output = self._liso_output(script)

            # compare output
            self.compare(output)

    def compare(self, liso_output):
        # frequencies
        frequencies = liso_output.frequencies

        # get LISO solution
        liso_solution = liso_output.solution()

        # run native
        native_solution = liso_output.circuit.solve(frequencies,
            output_nodes=list(liso_output.output_nodes))

        # check if they match
        self.assertEqual(liso_solution, native_solution)

    def _liso_output(self, script):
        # run LISO and parse output
        return Runner(script).run()

    @property
    def fil_scripts(self):
        return glob.glob(os.path.join(self.fil_dir, "*.fil"))

    @property
    def fil_dir(self):
        this_script_dir = os.path.dirname(os.path.realpath(__file__))

        return os.path.abspath(os.path.join(this_script_dir, self.REL_FIL_DIR))

if __name__ == '__main__':
    unittest.main()
