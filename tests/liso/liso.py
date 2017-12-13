import os.path
import glob
import numpy as np
import unittest

from electronics.simulate.liso import CircuitParser, Runner

class TestLiso(unittest.TestCase):
    REL_FIL_DIR = "."

    def test_scripts(self):
        for script in self.fil_scripts:
            output = self._liso_output(script)

            # compare output
            self.compare_liso(script, output)

    def compare_liso(self, script, liso_output):
        # frequencies
        liso_frequencies = liso_output.frequencies

        # parse LISO script
        parser = CircuitParser()
        parser.load(script)

        # run LISO
        parser.run()
        solution = parser.solution

        # asset frequencies are the same (they should be)
        self.assertTrue(np.allclose(liso_frequencies, solution.frequencies))

        # TODO: test actual data

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
