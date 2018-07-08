"""Component tests"""

from unittest import TestCase

from circuit.components import Component as ComponentBase, Resistor, Capacitor, Inductor, Node

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
