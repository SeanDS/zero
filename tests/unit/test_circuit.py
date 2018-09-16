"""Circuit tests"""

from unittest import TestCase

from circuit import Circuit
from circuit.components import Resistor, Capacitor, Inductor, OpAmp, Node

class CircuitTestCase(TestCase):
    """Circuit tests"""
    def setUp(self):
        self.reset()

    def reset(self):
        """Reset circuit"""
        self.circuit = Circuit()

    def test_add_component(self):
        """Test add component to circuit"""
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")
        l = Inductor(name="l1", value="1u", node1="n1", node2="n2")

        self.circuit.add_component(r)
        self.assertCountEqual(self.circuit.components, [r])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.assertEqual(self.circuit.n_components, 1)
        self.assertEqual(self.circuit.n_nodes, 2)
        self.circuit.add_component(c)
        self.assertCountEqual(self.circuit.components, [r, c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.assertEqual(self.circuit.n_components, 2)
        self.assertEqual(self.circuit.n_nodes, 2)
        self.circuit.add_component(l)
        self.assertCountEqual(self.circuit.components, [r, c, l])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.assertEqual(self.circuit.n_components, 3)
        self.assertEqual(self.circuit.n_nodes, 2)

    def test_add_component_without_name(self):
        """Test add component without name to circuit"""
        r = Resistor(value=10e3, node1="n1", node2="n2")
        c = Capacitor(value="10u", node1="n1", node2="n2")
        l = Inductor(value="1u", node1="n1", node2="n2")
        op = OpAmp(model="OP00", node1="n1", node2="n2", node3="n3")

        self.assertEqual(r.name, None)
        self.circuit.add_component(r)
        self.assertEqual(r.name, "r1")

        self.assertEqual(c.name, None)
        self.circuit.add_component(c)
        self.assertEqual(c.name, "c1")

        self.assertEqual(l.name, None)
        self.circuit.add_component(l)
        self.assertEqual(l.name, "l1")

        self.assertEqual(op.name, None)
        self.circuit.add_component(op)
        self.assertEqual(op.name, "op1")

    def test_remove_component(self):
        """Test remove component from circuit"""
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")

        self.circuit.add_component(r)
        self.circuit.add_component(c)

        self.assertCountEqual(self.circuit.components, [r, c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.assertEqual(self.circuit.n_components, 2)
        self.assertEqual(self.circuit.n_nodes, 2)

        self.circuit.remove_component(r)

        self.assertEqual(self.circuit.components, [c])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [Node("n1"), Node("n2")])
        self.assertEqual(self.circuit.n_components, 1)
        self.assertEqual(self.circuit.n_nodes, 2)

        self.circuit.remove_component(c)

        self.assertEqual(self.circuit.components, [])
        self.assertEqual(self.circuit.non_gnd_nodes, [])
        self.assertEqual(self.circuit.n_components, 0)
        self.assertEqual(self.circuit.n_nodes, 0)

    def test_cannot_add_component_with_invalid_name(self):
        """Test components with invalid names cannot be added to circuit"""
        # name "all" is invalid
        r = Resistor(name="all", value=1e3, node1="n1", node2="n2")
        self.assertRaisesRegex(ValueError, r"component name 'all' is reserved", self.circuit.add_component, r)

        # name "sum" is invalid
        c = Capacitor(name="sum", value=1e3, node1="n1", node2="n2")
        self.assertRaisesRegex(ValueError, r"component name 'sum' is reserved", self.circuit.add_component, c)

    def test_cannot_add_duplicate_component(self):
        """Test component already present in circuit cannot be added again"""
        # duplicate component
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        r2 = Resistor(name="r1", value=2e5, node1="n3", node2="n4")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"component with name 'r1' already in circuit",
                               self.circuit.add_component, r2)

        self.reset()

        # different component with same name
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        op1 = OpAmp(name="r1", model="OP00", node1="n3", node2="n4", node3="n5")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"component with name 'r1' already in circuit",
                               self.circuit.add_component, op1)
