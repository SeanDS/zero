"""Circuit tests"""

from unittest import TestCase

from circuit import Circuit
from circuit.components import Resistor, Capacitor, Inductor, Node

class CircuitTestCase(TestCase):
    def setUp(self):
        self.circuit = Circuit()

    def test_add_component(self):
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")
        l = Inductor(name="l1", value="1u", node1="n1", node2="n2")

        self.circuit.add_component(r)
        self.assertCountEqual(self.circuit.components, [r])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.circuit.add_component(c)
        self.assertCountEqual(self.circuit.components, [r, c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.circuit.add_component(l)
        self.assertCountEqual(self.circuit.components, [r, c, l])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

    def test_remove_component(self):
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")

        self.circuit.add_component(r)
        self.circuit.add_component(c)

        self.assertCountEqual(self.circuit.components, [r, c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        self.circuit.remove_component(r)

        self.assertEqual(self.circuit.components, [c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        self.circuit.remove_component(c)

        self.assertEqual(self.circuit.components, [])
        self.assertEqual(self.circuit.non_gnd_nodes, [])
    
    def test_add_invalid_component(self):
        # name "all" is invalid
        r = Resistor(name="all", value=1e3, node1="n1", node2="n2")
        self.assertRaisesRegex(ValueError, r"component name 'all' is reserved", self.circuit.add_component, r)

        # name "sum" is invalid
        c = Capacitor(name="sum", value=1e3, node1="n1", node2="n2")
        self.assertRaisesRegex(ValueError, r"component name 'sum' is reserved", self.circuit.add_component, c)