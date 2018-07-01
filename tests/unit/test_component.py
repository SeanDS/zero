"""Component tests"""

from unittest import TestCase

from circuit.components import Component as ComponentBase, Resistor, Capacitor, Inductor, Node

class Component(ComponentBase):
    """Child class of abstract ComponentBase"""
    
    def equation(self):
        raise NotImplementedError

class ComponentTestCase(TestCase):
    def setUp(self):
        pass

    def test_name_empty(self):
        # empty string
        name = ""
        component = Component(name=name)
        self.assertEqual(name, component.name)

        # null
        name = None
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_name_normal(self):
        name = "component_name"
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_nodes_empty(self):
        # empty list
        nodes = []
        component = Component(nodes=nodes)
        self.assertEqual(nodes, component.nodes)

        # null
        nodes = None
        component = Component(nodes=nodes)
        # default to empty list
        self.assertEqual([], component.nodes)
    
    def test_nodes_normal(self):
        nodes = [Node("n1"), Node("n2")]

        component = Component(nodes=nodes)
        self.assertEqual(nodes, component.nodes)

class PassiveTestCase(TestCase):
    def setUp(self):
        pass

    def test_set_value(self):
        r = Resistor(value=10e3, node1="n1", node2="n2")
        c = Capacitor(value="1.1n", node1="n1", node2="n2")
        l = Inductor(value="7.5u", node1="n1", node2="n2")

        # set values as floats
        r.resistance = 101e3
        c.capacitance = 101e3
        l.inductance = 101e3
        self.assertEquals(r.resistance, 101e3)
        self.assertEquals(c.capacitance, 101e3)
        self.assertEquals(l.inductance, 101e3)

        # set values as strings
        r.resistance = "101k"
        c.capacitance = "101k"
        l.inductance = "101k"
        self.assertEquals(r.resistance, 101e3)
        self.assertEquals(c.capacitance, 101e3)
        self.assertEquals(l.inductance, 101e3)