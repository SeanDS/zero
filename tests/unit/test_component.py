"""Component tests"""

from unittest import TestCase

from circuit.components import Component as ComponentBase

class Component(ComponentBase):
    """Child class of abstract ComponentBase"""
    
    def equation(self):
        return NotImplemented

class ComponentTestCase(TestCase):
    def setUp(self):
        pass

    def test_name(self):
        name = "component_name"
        component = Component(name=name)
        self.assertEqual(name, component.name)

    def test_name_empty(self):
        # empty string
        name = ""
        component = Component(name=name)
        self.assertEqual(name, component.name)

        # null
        name = None
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