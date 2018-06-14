import unittest
import numpy as np

from circuit.liso import LisoInputParser
from circuit.liso.base import LisoParserError
from circuit.components import Node

class LisoInputParserTestCase(unittest.TestCase):
    def setUp(self):
        self.reset()

    def reset(self):
        self.parser = LisoInputParser()

class ResistorTestCase(LisoInputParserTestCase):
    def test_resistor(self):
        self.parser.parse("r r1 10k n1 n2")
        r = self.parser.circuit.get_component("r1")
        self.assertEqual(r.name, "r1")
        self.assertEqual(r.resistance, 10e3)
        self.assertEqual(r.node1, Node("n1"))
        self.assertEqual(r.node2, Node("n2"))

class CapacitorTestCase(LisoInputParserTestCase):
    def test_capacitor(self):
        self.parser.parse("c c1 10n n1 n2")
        c = self.parser.circuit.get_component("c1")
        self.assertEqual(c.name, "c1")
        self.assertEqual(c.capacitance, 10e-9)
        self.assertEqual(c.node1, Node("n1"))
        self.assertEqual(c.node2, Node("n2"))

class InductorTestCase(LisoInputParserTestCase):
    def test_inductor(self):
        self.parser.parse("l l1 10u n1 n2")
        l = self.parser.circuit.get_component("l1")
        self.assertEqual(l.name, "l1")
        self.assertEqual(l.inductance, 10e-6)
        self.assertEqual(l.node1, Node("n1"))
        self.assertEqual(l.node2, Node("n2"))

class OpAmpTestCase(LisoInputParserTestCase):
    def test_opamp(self):
        self.parser.parse("op op1 OP00 n1 n2 n3")
        op = self.parser.circuit.get_component("op1")
        self.assertEqual(op.name, "op1")
        self.assertEqual(op.model.lower(), "op00")
        self.assertEqual(op.node1, Node("n1"))
        self.assertEqual(op.node2, Node("n2"))
        self.assertEqual(op.node3, Node("n3"))

    def test_invalid_model(self):
        self.assertRaisesRegex(ValueError, r"op-amp model __opinvalid__ not found in library",
                               self.parser.parse, "op op1 __opinvalid__ n1 n2 n3")

    def test_opamp_override(self):
        self.parser.parse("op op1 op27 n1 n2 n3 a0=123M")
        op = self.parser.circuit.get_component("op1")
        self.assertEqual(op.params["a0"], 123e6)

        self.parser.parse("op op2 ad797 n4 n5 n6 gbw=456k")
        op = self.parser.circuit.get_component("op2")
        self.assertEqual(op.params["gbw"], 456e3)

        self.parser.parse("op op3 lt1124 n4 n5 n6 sr=1G")
        op = self.parser.circuit.get_component("op3")
        self.assertEqual(op.params["sr"], 1e9)

    def test_opamp_invalid_override(self):
        # invalid scale
        kwargs = {"text": """
r r1 430 n1 nm
op op1 op27 np nm nout a1=123e6
c c1 10u gnd n1
"""}

        self.assertRaisesRegex(LisoParserError, r"unknown op-amp override parameter 'a1' \(line 3\)", self.parser.parse, **kwargs)

class FrequencyTestCase(LisoInputParserTestCase):
    def test_frequencies(self):
        self.parser.parse("freq lin 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.linspace(1e-1, 1e5, 1001)))

        self.reset()

        self.parser.parse("freq log 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.logspace(np.log10(1e-1), np.log10(1e5), 1001)))

    def test_invalid_scale(self):
        # invalid scale
        kwargs = {"text": """
r r1 430 n1 nm
freq dec 1 1M 1234
c c1 10u gnd n1
"""}

        self.assertRaisesRegex(LisoParserError, r"invalid frequency scale 'dec' \(line 3\)", self.parser.parse, **kwargs)
    
    def test_cannot_redefine_frequencies(self):
        self.parser.parse("freq lin 0.1 100k 1000")
        # try to set frequencies again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine frequencies \(line 2\)", self.parser.parse, "freq lin 0.1 100k 1000")

class InputTestCase(LisoInputParserTestCase):
    def test_input(self):
        self.parser.parse("uinput nin")
        self.assertEqual(self.parser.input_type, "voltage")

        self.reset()

        self.parser.parse("iinput nin")
        self.assertEqual(self.parser.input_type, "current")

    def test_cannot_redefine_input_type(self):
        self.parser.parse("uinput nin")
        # try to set input again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine input type \(line 2\)", self.parser.parse, "uinput nin")

class NoiseOutputNodeTestCase(LisoInputParserTestCase):
    def test_noise(self):
        self.parser.parse("noise nout")
        self.assertEqual(self.parser.noise_output_node, Node("nout"))

    def test_cannot_redefine_noise_node(self):
        self.parser.parse("noise nout")
        # try to set noise node again
        self.assertRaisesRegex(LisoParserError, r"cannot redefine noise output node \(line 2\)", self.parser.parse, "noise nin")

class SyntaxErrorTestCase(LisoInputParserTestCase):
    """Syntax error tests that don't fit into individual components or commands"""

    def test_invalid_component(self):
        # component type "a" doesn't exist
        kwargs = {"text": """
a c1 10u gnd n1
r r1 430 n1 nm
"""}

        self.assertRaisesRegex(LisoParserError, r"'a' \(line 2\)", self.parser.parse, **kwargs)

    def test_missing_component_name(self):
        # no component name given
        kwargs = {"text": """
c 10u gnd n1
r r1 430 n1 nm
"""}

        self.assertRaisesRegex(LisoParserError, r"unexpected end of line \(line 2\)", self.parser.parse, **kwargs)

    def test_invalid_component_value(self):
        # invalid component value
        kwargs = {"text": """
r r1 430 n1 nm
c c1 -10u gnd n1
"""}

        self.assertRaisesRegex(LisoParserError, r"illegal character '-' \(line 3, position 5\)", self.parser.parse, **kwargs)

    def test_invalid_component_node(self):
        # invalid component value
        kwargs = {"text": """
r r1 430 n1 nm
c c1 10u gnd @
"""}

        self.assertRaisesRegex(LisoParserError, r"illegal character '@' \(line 3, position 13\)", self.parser.parse, **kwargs)
    
    def test_duplicate_component(self):
        # duplicate component
        kwargs = {"text": """
r r1 10k n1 n2
r r1 10k n1 n2
"""}

        self.assertRaisesRegex(ValueError, r"component with name 'r1' already in circuit", self.parser.parse, **kwargs)

        self.reset()

        # different component with same name
        kwargs = {"text": """
r r1 10k n1 n2
op r1 op00 n1 n2 n3
"""}

        self.assertRaisesRegex(ValueError, r"component with name 'r1' already in circuit", self.parser.parse, **kwargs)