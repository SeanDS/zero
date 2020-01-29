"""Circuit tests"""

from unittest import TestCase
from itertools import permutations
from copy import deepcopy
import numpy as np

from zero import Circuit
from zero.components import Resistor, Capacitor, Inductor, OpAmp, Node
from zero.analysis import AcSignalAnalysis
from ..data import ZeroDataTestCase


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

    def test_remove_component_by_name(self):
        """Test remove component from circuit by name"""
        r = Resistor(name="r1", value=10e3, node1="n1", node2="n2")
        c = Capacitor(name="c1", value="10u", node1="n1", node2="n2")

        self.circuit.add_component(r)
        self.circuit.add_component(c)

        self.circuit.remove_component("r1")
        self.circuit.remove_component("c1")

        self.assertEqual(self.circuit.components, [])
        self.assertCountEqual(self.circuit.non_gnd_nodes, [])
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
        self.assertRaisesRegex(ValueError, r"element with name 'r1' already in circuit",
                               self.circuit.add_component, r2)

        self.reset()

        # different component with same name
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        op1 = OpAmp(name="r1", model="OP00", node1="n3", node2="n4", node3="n5")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"element with name 'r1' already in circuit",
                               self.circuit.add_component, op1)

    def test_cannot_add_component_with_same_name_as_node(self):
        """Test component with same name as existing node cannot be added"""
        # first component
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        # second component, with same name as one of first component's nodes
        r2 = Resistor(name="n1", value=2e5, node1="n3", node2="r4")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"element with name 'n1' already in circuit",
                               self.circuit.add_component, r2)

        self.reset()

        # different component with same name
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        op1 = OpAmp(name="n1", model="OP00", node1="n2", node2="n4", node3="n5")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"element with name 'n1' already in circuit",
                               self.circuit.add_component, op1)

    def test_cannot_add_node_with_same_name_as_component(self):
        """Test node with same name as existing component cannot be added"""
        # first component
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        # second component, with node with same name as first component
        r2 = Resistor(name="r2", value=2e5, node1="n3", node2="r1")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"node 'r1' is the same as existing circuit component",
                               self.circuit.add_component, r2)

        self.reset()

        # different component with same name
        r1 = Resistor(name="r1", value=1e3, node1="n1", node2="n2")
        op1 = OpAmp(name="op1", model="OP00", node1="r1", node2="n4", node3="n5")

        self.circuit.add_component(r1)
        self.assertRaisesRegex(ValueError, r"node 'r1' is the same as existing circuit component",
                               self.circuit.add_component, op1)


class TestComponentReplacement(ZeroDataTestCase):
    def setUp(self):
        super().setUp()
        self.passives = [self._resistor(), self._resistor(),
                         self._capacitor(), self._capacitor(),
                         self._inductor(), self._inductor()]

    def test_replace_passive_passive(self):
        """Test passive component replacement."""
        for cmp1, cmp2 in permutations(self.passives, 2):
            with self.subTest((cmp1, cmp2)):
                circuit = Circuit()
                circuit.add_component(cmp1)
                self.assertTrue(circuit.has_component(cmp1.name))
                circuit.replace_component(cmp1, cmp2)
                # Test circuit composition.
                self.assertTrue(circuit.has_component(cmp2.name))
                self.assertFalse(circuit.has_component(cmp1.name))
                # Nodes in cmp2 should have been copied from cmp1.
                self.assertEqual(cmp1.nodes, cmp2.nodes)

    def test_replace_opamp_opamp(self):
        """Test op-amp replacement."""
        op1 = self._opamp()
        op2 = self._opamp()
        circuit = Circuit()
        circuit.add_component(op1)
        self.assertTrue(circuit.has_component(op1.name))
        circuit.replace_component(op1, op2)
        # Test circuit composition.
        self.assertTrue(circuit.has_component(op2.name))
        self.assertFalse(circuit.has_component(op1.name))
        # Nodes in cmp2 should have been copied from cmp1.
        self.assertEqual(op1.nodes, op2.nodes)

    def test_cannot_replace_passive_opamp_or_opamp_passive(self):
        """Test passive components cannot be replaced with op-amps (and vice versa)."""
        opamp = self._opamp()
        for passive in self.passives:
            # Test replacement of op-amp.
            with self.subTest((opamp, passive)):
                circuit = Circuit()
                circuit.add_component(opamp)
                self.assertTrue(circuit.has_component(opamp.name))
                self.assertRaises(ValueError, circuit.replace_component, opamp, passive)
                # Test that the circuit still has the op-amp.
                self.assertTrue(circuit.has_component(opamp.name))
            # Test replacement of passive.
            with self.subTest((passive, opamp)):
                circuit = Circuit()
                circuit.add_component(passive)
                self.assertTrue(circuit.has_component(passive.name))
                self.assertRaises(ValueError, circuit.replace_component, passive, opamp)
                # Test that the circuit still has the passive.
                self.assertTrue(circuit.has_component(passive.name))


class TestCircuitCopying(ZeroDataTestCase):
    def test_deep_copy(self):
        """Test deep copying of a circuit."""
        circuit1 = Circuit()
        circuit1.add_resistor(name="R1", value=50, node1="n1", node2="n2")
        circuit1.add_resistor(name="R2", value=50, node1="n2", node2="gnd")

        frequencies = np.logspace(0, 3, 11)

        analysis1 = AcSignalAnalysis(circuit=circuit1)
        sol1 = analysis1.calculate(frequencies=frequencies, input_type="voltage", node="n1")

        # Copy the circuit.
        circuit2 = deepcopy(circuit1)

        analysis2 = AcSignalAnalysis(circuit=circuit2)
        sol2 = analysis2.calculate(frequencies=frequencies, input_type="voltage", node="n1")

        self.assertTrue(sol1.equivalent_to(sol2))
