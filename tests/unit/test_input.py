"""Input component tests"""

from unittest import TestCase
from zero.components import Input, Node


class InputTestCase(TestCase):
    """Test circuit input component"""
    def test_signal_input(self):
        """Test input component name, nodes, etc. after construction"""
        for input_type in ("voltage", "current"):
            with self.subTest(input_type):
                inpt = Input(input_type=input_type, nodes=["gnd", "nin"])
                self.assertEqual(inpt.name, "input")
                self.assertEqual(inpt.input_type, input_type)
                self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])
                self.assertEqual(inpt.node1, Node("gnd"))
                self.assertEqual(inpt.node2, Node("nin"))
                self.assertEqual(inpt.node_n, inpt.node1)
                self.assertEqual(inpt.node_p, inpt.node2)

    def test_noise_input(self):
        """Test input component name, nodes, etc. after construction"""
        for input_type in ("voltage", "current"):
            with self.subTest(input_type):
                inpt = Input(input_type=input_type, nodes=["gnd", "nin"], is_noise=True,
                             impedance="15.5k")
                self.assertEqual(inpt.name, "input")
                self.assertEqual(inpt.input_type, input_type)
                self.assertEqual(inpt.impedance, 15.5e3)
                self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])
                self.assertEqual(inpt.node1, Node("gnd"))
                self.assertEqual(inpt.node2, Node("nin"))
                self.assertEqual(inpt.node_n, inpt.node1)
                self.assertEqual(inpt.node_p, inpt.node2)

    def test_name_cannot_be_set(self):
        """Test cannot set input component name"""
        for input_type in ("voltage", "current"):
            with self.subTest(input_type):
                self.assertRaises(TypeError, Input, name="abc", input_type=input_type,
                                  nodes=["gnd", "nin"])

    def test_voltage_input(self):
        """Test voltage input"""
        inpt = Input(input_type="voltage", nodes=["gnd", "nin"])
        # no input impedance
        self.assertEqual(inpt.impedance, None)

    def test_current_input(self):
        """Test current input"""
        inpt = Input(input_type="current", nodes=["gnd", "nin"])
        # input impedance should be None
        self.assertEqual(inpt.impedance, None)
