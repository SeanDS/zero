"""Circuit tests"""

from unittest import TestCase

from circuit import Circuit
from circuit.components import Resistor, Capacitor, Node

class CircuitTestCase(TestCase):
    def setUp(self):
        pass

    def test_remove_component(self):
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")
        circuit = Circuit()
        circuit.add_component(r)
        circuit.add_component(c)

        self.assertEqual(circuit.components, [r, c])
        self.assertEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        circuit.remove_component(r)

        self.assertEqual(circuit.components, [c])
        self.assertEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        circuit.remove_component(c)

        self.assertEqual(circuit.components, [])
        self.assertEqual(circuit.non_gnd_nodes, [])