import unittest
import numpy as np

from circuit.liso import LisoInputParser
from circuit.components import Node

class LisoInputParserTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = LisoInputParser()

    def test_resistor(self):
        self.parser.parse("r r1 10k n1 n2")
        r = self.parser.circuit.get_component("r1")
        self.assertEqual(r.name, "r1")
        self.assertEqual(r.resistance, 10e3)
        self.assertEqual(r.node1, Node("n1"))
        self.assertEqual(r.node2, Node("n2"))

    def test_capacitor(self):
        self.parser.parse("c c1 10n n1 n2")
        c = self.parser.circuit.get_component("c1")
        self.assertEqual(c.name, "c1")
        self.assertEqual(c.capacitance, 10e-9)
        self.assertEqual(c.node1, Node("n1"))
        self.assertEqual(c.node2, Node("n2"))

    def test_inductor(self):
        self.parser.parse("l l1 10u n1 n2")
        l = self.parser.circuit.get_component("l1")
        self.assertEqual(l.name, "l1")
        self.assertEqual(l.inductance, 10e-6)
        self.assertEqual(l.node1, Node("n1"))
        self.assertEqual(l.node2, Node("n2"))

    def test_opamp(self):
        self.parser.parse("op op1 OP00 n1 n2 n3")
        op = self.parser.circuit.get_component("op1")
        self.assertEqual(op.name, "op1")
        self.assertEqual(op.model.lower(), "op00")
        self.assertEqual(op.node1, Node("n1"))
        self.assertEqual(op.node2, Node("n2"))
        self.assertEqual(op.node3, Node("n3"))

    def test_lin_frequencies(self):
        self.parser.parse("freq lin 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.linspace(1e-1, 1e5, 1001)))

    def test_log_frequencies(self):
        self.parser.parse("freq log 0.1 100k 1000")
        self.assertTrue(np.allclose(self.parser.frequencies, np.logspace(np.log10(1e-1), np.log10(1e5), 1001)))

class LisoInputParserSyntaxErrorTestCase(unittest.TestCase):
    """Syntax error tests"""

    def setUp(self):
        self.parser = LisoInputParser()

    def test_component_invalid_type(self):
        # component type "a" doesn't exist
        kwargs = {"text": """
a c1 10u gnd n1
r r1 430 n1 nm
"""}

        self.assertRaisesRegex(SyntaxError, r"'a' at line 2", self.parser.parse, **kwargs)

    def test_component_missing_name(self):
        # no component name given
        kwargs = {"text": """
c 10u gnd n1
r r1 430 n1 nm
"""}

        self.assertRaisesRegex(SyntaxError, r"unexpected end of line on line 3", self.parser.parse, **kwargs)

    def test_component_invalid_value(self):
        # invalid component value
        kwargs = {"text": """
r r1 430 n1 nm
c c1 -10u gnd n1
"""}

        self.assertRaisesRegex(SyntaxError, r"illegal character '-' on line 3 at position 5", self.parser.parse, **kwargs)

    def test_component_invalid_node(self):
        # invalid component value
        kwargs = {"text": """
r r1 430 n1 nm
c c1 10u gnd @
"""}

        self.assertRaisesRegex(SyntaxError, r"illegal character '@' on line 3 at position 13", self.parser.parse, **kwargs)