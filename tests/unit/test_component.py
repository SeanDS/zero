"""Component tests"""

from unittest import TestCase

from zero.components import Component as ComponentBase, Resistor, Capacitor, Inductor, Node


class Component(ComponentBase):
    """Child class of abstract ComponentBase, used to test :class:`components <.Component>` \
       directly"""
    def equation(self):
        """Mock component equation (not needed for tests)"""
        raise NotImplementedError


class ComponentTestCase(TestCase):
    """Component tests"""
    def test_name_normal(self):
        """Test creating component with normal name"""
        name = "component_name"
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_name_empty(self):
        """Test creating component with empty name"""
        name = ""
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_name_none(self):
        """Test creating component with NoneType name"""
        # null
        name = None
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_nodes_normal(self):
        """Test creating component with node list"""
        nodes = [Node("n1"), Node("n2")]

        component = Component(nodes=nodes)
        self.assertEqual(nodes, component.nodes)

    def test_nodes_empty(self):
        """Test creating component with empty node list"""
        nodes = []
        component = Component(nodes=nodes)
        self.assertEqual(nodes, component.nodes)

    def test_nodes_none(self):
        """Test creating component with NoneType node list"""
        nodes = None
        component = Component(nodes=nodes)
        # default to empty list
        self.assertEqual([], component.nodes)


class PassiveTestCase(TestCase):
    """Passive component tests"""
    def test_set_value(self):
        """Test set passive component values"""
        r = Resistor(value=10e3, node1="n1", node2="n2")
        c = Capacitor(value="1.1n", node1="n1", node2="n2")
        l = Inductor(value="7.5u", node1="n1", node2="n2")

        # set values as floats
        r.resistance = 101e3
        c.capacitance = 101e3
        l.inductance = 101e3
        self.assertEqual(r.resistance, 101e3)
        self.assertEqual(c.capacitance, 101e3)
        self.assertEqual(l.inductance, 101e3)

        # set values as strings
        r.resistance = "101k"
        c.capacitance = "101k"
        l.inductance = "101k"
        self.assertEqual(r.resistance, 101e3)
        self.assertEqual(c.capacitance, 101e3)
        self.assertEqual(l.inductance, 101e3)


class InductorTestCase(TestCase):
    """Inductor component tests"""
    def test_coupling_factor(self):
        """Test set invalid coupling factor"""
        l1 = Inductor(value="10u", node1="n1", node2="n2")
        l2 = Inductor(value="40u", node1="n3", node2="n4")

        # valid
        l1.coupling_factors[l2] = 0
        l2.coupling_factors[l1] = 0
        l1.coupling_factors[l2] = 0.5
        l2.coupling_factors[l1] = 0.5
        l1.coupling_factors[l2] = 1
        l2.coupling_factors[l1] = 1

        # coupled inductors list should contain other inductor
        self.assertCountEqual(l1.coupled_inductors, [l2])
        self.assertCountEqual(l2.coupled_inductors, [l1])

        # cannot set coupling factors out of range
        self.assertRaises(ValueError, l1.coupling_factors.__setitem__, l2, -0.5)
        self.assertRaises(ValueError, l1.coupling_factors.__setitem__, l2, 1.1)

    def test_coupling_factor_invalid_other(self):
        """Test set invalid coupling component"""
        l1 = Inductor(value="10u", node1="n1", node2="n2")
        r1 = Resistor(value="10k", node1="n3", node2="n4")

        # cannot couple to a resistor
        self.assertRaises(TypeError, l1.coupling_factors.__setitem__, r1, 1)

    def test_mutual_inductance(self):
        """Test set coupling factor between two inductors"""
        l1 = Inductor(value="10u", node1="n1", node2="n2")
        l2 = Inductor(value="40u", node1="n3", node2="n4")
        l3 = Inductor(value="80u", node1="n5", node2="n6")

        l1.coupling_factors[l2] = 0.95
        l2.coupling_factors[l1] = 0.95

        self.assertAlmostEqual(l1.inductance_from(l2), 0.000019)
        self.assertAlmostEqual(l2.inductance_from(l1), 0.000019)

        l1.coupling_factors[l2] = 0.5
        l2.coupling_factors[l1] = 0.5

        self.assertAlmostEqual(l1.inductance_from(l2), 0.00001)
        self.assertAlmostEqual(l2.inductance_from(l1), 0.00001)

        l1.coupling_factors[l2] = 0
        l2.coupling_factors[l1] = 0

        self.assertEqual(l1.inductance_from(l2), 0)
        self.assertEqual(l2.inductance_from(l1), 0)

        # mutual inductance to inductor where coupling hasn't been set is still 0
        self.assertEqual(l1.inductance_from(l3), 0)
        self.assertEqual(l3.inductance_from(l1), 0)
