"""Circuit tests"""

from unittest import TestCase

from circuit import Circuit
from circuit.components import Resistor, Capacitor, Inductor, Node

class CircuitTestCase(TestCase):
    def setUp(self):
        pass

    def test_add_component(self):
        circuit = Circuit()
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")
        l = Inductor(name="l1", value="1u", node1="n1", node2="n2")

        circuit.add_component(r)
        self.assertCountEqual(circuit.components, [r])
        self.assertCountEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        circuit.add_component(c)
        self.assertCountEqual(circuit.components, [r, c])
        self.assertCountEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        circuit.add_component(l)
        self.assertCountEqual(circuit.components, [r, c, l])
        self.assertCountEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

    def test_remove_component(self):
        circuit = Circuit()
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")

        circuit.add_component(r)
        circuit.add_component(c)

        self.assertCountEqual(circuit.components, [r, c])
        self.assertCountEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        circuit.remove_component(r)

        self.assertEqual(circuit.components, [c])
        self.assertCountEqual(circuit.non_gnd_nodes, [Node("n1"), Node("n2")])

        circuit.remove_component(c)

        self.assertEqual(circuit.components, [])
        self.assertEqual(circuit.non_gnd_nodes, [])