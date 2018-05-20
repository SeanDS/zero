"""Component tests"""

from unittest import TestCase

from circuit.components import Component

def concreter(_class, *args, **kwargs):
    """Creates a non-abstract version of an abstract class
    
    See https://stackoverflow.com/a/9759329/2251982.
    """

    if not "__abstractmethods__" in _class.__dict__:
        return _class

    new_dict = _class.__dict__.copy()

    for abstractmethod in _class.__abstractmethods__:
        # replace each abstract method or property with an identity function
        new_dict[abstractmethod] = lambda x, *args, **kwargs: (x, args, kwargs)

    # create a new class, overriding the abstract base class
    return type("dummy_concrete_%s" % _class.__name__, (_class,), new_dict)

component_class = concreter(Component)

class ComponentTestCase(TestCase):
    def setUp(self):
        pass

    def test_name(self):
        name = "component_name"
        component = component_class(name=name)
        self.assertEqual(name, component.name)

    def test_name_empty(self):
        # empty string
        name = ""
        component = component_class(name=name)
        self.assertEqual(name, component.name)

        # null
        name = None
        component = component_class(name=name)
        self.assertEqual(name, component.name)

    def test_nodes_empty(self):
        # empty list
        nodes = []
        component = component_class(nodes=nodes)
        self.assertEqual(nodes, component.nodes)

        # null
        nodes = None
        component = component_class(nodes=nodes)
        # default to empty list
        self.assertEqual([], component.nodes)