import unittest
import numpy as np

from circuit.liso import LisoInputParser, LisoParserError

class LisoInputParserTestCase(unittest.TestCase):
    def setUp(self):
        self.reset()

    def reset(self):
        self.parser = LisoInputParser()

class VoltageOutputTestCase(LisoInputParserTestCase):
    def test_invalid_output_node(self):
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# n3 doesn't exist
uoutput n3
""")
        self.assertRaisesRegex(LisoParserError,
                               r"output element 'n3' is not present in the circuit",
                               self.parser.solution)

class CurrentOutputTestCase(LisoInputParserTestCase):
    def test_invalid_output_node(self):
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# r2 doesn't exist
ioutput r2
""")
        self.assertRaisesRegex(LisoParserError,
                               r"output element 'r2' is not present in the circuit",
                               self.parser.solution)

class NoiseOutputTestCase(LisoInputParserTestCase):
    def test_invalid_noise_output_node(self):
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# n3 doesn't exist
noise n3 all
""")
        self.assertRaisesRegex(LisoParserError,
                               r"noise output node 'n3' is not present in the circuit",
                               self.parser.solution)
