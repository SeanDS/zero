"""LISO input parser tests"""

import unittest
import numpy as np

from zero.liso import LisoInputParser, LisoParserError
from zero.components import Node, NoiseNotFoundError


class LisoInputParserTestCase(unittest.TestCase):
    def setUp(self):
        self.reset()

    def reset(self):
        self.parser = LisoInputParser()


class ResistorTestCase(LisoInputParserTestCase):
    def test_resistor(self):
        self.parser.parse("r r1 10k n1 n2")
        r = self.parser.circuit["r1"]
        self.assertEqual(r.name, "r1")
        self.assertAlmostEqual(r.resistance, 10e3)
        self.assertEqual(r.node1, Node("n1"))
        self.assertEqual(r.node2, Node("n2"))


class CapacitorTestCase(LisoInputParserTestCase):
    def test_capacitor(self):
        self.parser.parse("c c1 10n n1 n2")
        c = self.parser.circuit["c1"]
        self.assertEqual(c.name, "c1")
        self.assertAlmostEqual(c.capacitance, 10e-9)
        self.assertEqual(c.node1, Node("n1"))
        self.assertEqual(c.node2, Node("n2"))


class InductorTestCase(LisoInputParserTestCase):
    def test_inductor(self):
        self.parser.parse("l l1 10u n1 n2")
        l = self.parser.circuit["l1"]
        self.assertEqual(l.name, "l1")
        self.assertAlmostEqual(l.inductance, 10e-6)
        self.assertEqual(l.node1, Node("n1"))
        self.assertEqual(l.node2, Node("n2"))


class OpAmpTestCase(LisoInputParserTestCase):
    def test_opamp(self):
        self.parser.parse("op op1 OP00 n1 n2 n3")
        op = self.parser.circuit["op1"]
        self.assertEqual(op.name, "op1")
        self.assertEqual(op.model.lower(), "op00")
        self.assertEqual(op.node1, Node("n1"))
        self.assertEqual(op.node2, Node("n2"))
        self.assertEqual(op.node3, Node("n3"))

    def test_invalid_model(self):
        self.assertRaisesRegex(ValueError, r"op-amp model '__opinvalid__' not found in library",
                               self.parser.parse, "op op1 __opinvalid__ n1 n2 n3")

    def test_opamp_override(self):
        self.parser.parse("op op1 op27 n1 n2 n3 a0=123M")
        op = self.parser.circuit["op1"]
        self.assertAlmostEqual(op.params["a0"], 123e6)

        self.parser.parse("op op2 ad797 n4 n5 n6 a0=123M gbw=456k")
        op = self.parser.circuit["op2"]
        self.assertAlmostEqual(op.params["a0"], 123e6)
        self.assertAlmostEqual(op.params["gbw"], 456e3)

        self.parser.parse("op op3 lt1124 n4 n5 n6 a0=123M gbw=456k sr=1G")
        op = self.parser.circuit["op3"]
        self.assertAlmostEqual(op.params["a0"], 123e6)
        self.assertAlmostEqual(op.params["gbw"], 456e3)
        self.assertAlmostEqual(op.params["sr"], 1e9)

    def test_opamp_invalid_override(self):
        # invalid scale
        text = """
r r1 430 n1 nm
op op1 op27 np nm nout a1=123e6
c c1 10u gnd n1
"""

        self.assertRaisesRegex(LisoParserError,
                               r"unknown op-amp override parameter 'a1' \(line 3\)",
                               self.parser.parse, text=text)


class FrequencyTestCase(LisoInputParserTestCase):
    def test_frequencies(self):
        self.parser.parse("freq lin 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.linspace(1e-1, 1e5, 1001)))

        self.reset()

        self.parser.parse("freq log 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.logspace(np.log10(1e-1),
                                    np.log10(1e5), 1001)))

        self.reset()

        self.parser.parse("freq lin 1e-3 1e5 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.linspace(1e-3, 1e5, 1001)))

    def test_invalid_scale(self):
        # invalid scale
        text = """
r r1 430 n1 nm
freq dec 1 1M 1234
c c1 10u gnd n1
"""

        self.assertRaisesRegex(LisoParserError, r"invalid frequency scale 'dec' \(line 3\)",
                               self.parser.parse, text)

    def test_cannot_redefine_frequencies(self):
        self.parser.parse("freq lin 0.1 100k 1000")
        # try to set frequencies again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine frequencies \(line 2\)",
                               self.parser.parse, "freq lin 0.1 100k 1000")


class VoltageInputTestCase(LisoInputParserTestCase):
    def test_input(self):
        self.parser.parse("uinput nin")
        self.assertEqual(self.parser.input_type, "voltage")
        self.assertEqual(self.parser.input_node_p, Node("nin"))
        self.assertEqual(self.parser.input_node_n, None)

    def test_cannot_redefine_input_type(self):
        self.parser.parse("uinput nin")
        # try to set input again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine input type \(line 2\)",
                               self.parser.parse, "uinput nin")

    def test_impedance(self):
        # defaults to 50 ohm
        self.parser.parse("uinput nin")
        self.assertEqual(self.parser.input_impedance, 50)

        self.reset()

        # unit parsing
        self.parser.parse("uinput nin 10M")
        self.assertAlmostEqual(self.parser.input_impedance, 10e6)

        self.reset()

        self.parser.parse("uinput nin 1e3")
        self.assertAlmostEqual(self.parser.input_impedance, 1e3)


class CurrentInputTestCase(LisoInputParserTestCase):
    def test_input(self):
        self.parser.parse("iinput nin")
        self.assertEqual(self.parser.input_type, "current")
        self.assertEqual(self.parser.output_type, None)
        self.assertEqual(self.parser.input_node_p, Node("nin"))
        self.assertEqual(self.parser.input_node_n, None)

    def test_cannot_redefine_input_type(self):
        self.parser.parse("iinput nin")
        # try to set input again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine input type \(line 2\)",
                               self.parser.parse, "iinput nin")

    def test_impedance(self):
        # defaults to 50 ohm
        self.parser.parse("iinput nin")
        self.assertEqual(self.parser.input_impedance, 50)

        self.reset()

        # unit parsing
        self.parser.parse("iinput nin 10k")
        self.assertAlmostEqual(self.parser.input_impedance, 10e3)

        self.reset()

        self.parser.parse("iinput nin 1e3")
        self.assertAlmostEqual(self.parser.input_impedance, 1e3)


class NoiseOutputNodeTestCase(LisoInputParserTestCase):
    def test_noise(self):
        self.parser.parse("noise nout n1")
        self.assertEqual(self.parser.output_type, "noise")
        self.assertEqual(self.parser.noise_output_element, "nout")

    def test_noise_suffices(self):
        text = """
r r1 1k n1 n3
r r2 10k n3 n4
r r3 10k n2 gnd
op op1 op00 n2 n3 n4
"""

        self.parser.parse(text)
        self.parser.parse("noise nout op1:u")
        self.assertEqual(len(self.parser.displayed_noise_objects), 1)

        self.reset()

        self.parser.parse(text)
        self.parser.parse("noise nout op1:u+-")
        self.assertEqual(len(self.parser.displayed_noise_objects), 3)

        self.reset()

        self.parser.parse(text)
        self.parser.parse("noise nout op1:-u+")
        self.assertEqual(len(self.parser.displayed_noise_objects), 3)

    def test_cannot_redefine_noise_node(self):
        self.parser.parse("noise nout n1")
        # try to set noise node again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine noise output element \(line 2\)",
                               self.parser.parse, "noise nin n1")

    def test_must_set_noise_output_element(self):
        # sink element defined, but no sources
        self.assertRaisesRegex(LisoParserError, r"unexpected end of file \(line 1\)",
                               self.parser.parse, "noise nout")

    def test_cannot_set_scaling_for_non_opamp(self):
        text = """
r r1 1k n1 n2
r r2 10k n2 n3
op op1 op00 gnd n2 n3
"""

        # try to set resistor voltage noise
        self.parser.parse(text)
        self.parser.parse("noise n3 r2:u")
        self.assertRaisesRegex(LisoParserError,
                               r"noise suffices cannot be specified on non-op-amps \(line 6\)",
                               getattr, self.parser, "displayed_noise_objects")

        self.reset()

        # try to set resistor non-inverting current noise
        self.parser.parse(text)
        self.parser.parse("noise n3 r2:+")
        self.assertRaisesRegex(LisoParserError,
                               r"noise suffices cannot be specified on non-op-amps \(line 6\)",
                               getattr, self.parser, "displayed_noise_objects")

        self.reset()

        # try to set resistor inverting current noise
        self.parser.parse(text)
        self.parser.parse("noise n3 r2:-")
        self.assertRaisesRegex(LisoParserError,
                               r"noise suffices cannot be specified on non-op-amps \(line 6\)",
                               getattr, self.parser, "displayed_noise_objects")

    def test_cannot_set_noisy_sum(self):
        text = """
r r1 1k n1 n2
r r2 10k n2 n3
op op1 op00 gnd n2 n3
"""

        self.parser.parse(text)
        self.parser.parse("noisy sum")
        self.assertRaisesRegex(LisoParserError,
                               r"cannot specify 'sum' as noisy source \(line 6\)",
                               getattr, self.parser, "summed_noise_objects")


class SyntaxErrorTestCase(LisoInputParserTestCase):
    """Syntax error tests that don't fit into individual components or commands"""

    def test_invalid_component(self):
        # component type "a" doesn't exist
        text = """
a c1 10u gnd n1
r r1 430 n1 nm
"""

        self.assertRaisesRegex(LisoParserError, r"'a' \(line 2\)", self.parser.parse, text)

    def test_missing_component_name(self):
        # no component name given
        text_1 = """
c 10u gnd n1
r r1 430 n1 nm
"""

        self.assertRaisesRegex(LisoParserError, r"unexpected end of line \(line 2\)",
                               self.parser.parse, text_1)

        self.reset()

        # no component name given, extra newline
        text_2 = """
c 10u gnd n1

r r1 430 n1 nm
"""

        self.assertRaisesRegex(LisoParserError, r"unexpected end of line \(line 2\)",
                               self.parser.parse, text_2)

    def test_invalid_component_value(self):
        # invalid component value
        text = """
r r1 430 n1 nm
c c1 -10u gnd n1
"""

        self.assertRaisesRegex(LisoParserError, r"illegal character '-' \(line 3, position 5\)",
                               self.parser.parse, text)

    def test_invalid_component_node(self):
        # invalid component value
        text = """
r r1 430 n1 nm
c c1 10u gnd @
"""

        self.assertRaisesRegex(LisoParserError, r"illegal character '@' \(line 3, position 13\)",
                               self.parser.parse, text)
