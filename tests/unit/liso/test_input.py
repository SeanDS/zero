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

    def test_component_invalid_type(self):
        kwargs = {"text": """
# component type "a" doesn't exist
a c1 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)
        self.assertRaisesRegex(SyntaxError, r"LISO syntax error 'a' at line 3", self.parser.parse, **kwargs)

    def test_component_missing_name(self):
        kwargs = {"text": """
# no component name given
c 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)

    def test_component_invalid_value(self):
        kwargs = {"text": """
# invalid component value
c c1 -10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)