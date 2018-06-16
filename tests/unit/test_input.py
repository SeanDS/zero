"""Input component tests"""

from unittest import TestCase

from circuit.components import Input, Node

class InputTestCase(TestCase):
    def setUp(self):
        pass

    def test_name(self):
        inpt = Input(input_type="noise", node="nin", impedance=50)
        self.assertEqual(inpt.name, "input")

    def test_name_cannot_be_set(self):
        """Input name is always 'input'"""
        self.assertRaises(ValueError, Input, name="abc", input_type="noise", node="nin")

    def test_nodes(self):
        inpt = Input(input_type="noise", node="nin", impedance=50)
        self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])

        inpt = Input(input_type="noise", node_n="nin", node_p="gnd", impedance=50)
        self.assertEqual(inpt.nodes, [Node("nin"), Node("gnd")])

        inpt = Input(input_type="noise", node_n="gnd", node_p="nin", impedance=50)
        self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])

    def test_node_properties(self):
        inpt = Input(input_type="noise", node_n="gnd", node_p="nin", impedance=50)
        self.assertEqual(inpt.node1, Node("gnd"))
        self.assertEqual(inpt.node2, Node("nin"))
        self.assertEqual(inpt.node_n, inpt.node1)
        self.assertEqual(inpt.node_p, inpt.node2)

    def test_invalid_node_combinations(self):
        # node and node_p
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n", Input,
                               input_type="noise", node="nin", node_p="nin", impedance=50)

        # node and node_n
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n", Input,
                               input_type="noise", node="nin", node_n="nin", impedance=50)

        # node and node_n and node_p
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n", Input,
                               input_type="noise", node="nin", node_n="nin", node_p="nin", impedance=50)

        # node_n but not node_p
        self.assertRaisesRegex(ValueError, r"node_p and node_n must both be specified", Input,
                               input_type="noise", node_n="nin", impedance=50)

        # node_p but not node_n
        self.assertRaisesRegex(ValueError, r"node_p and node_n must both be specified", Input,
                               input_type="noise", node_p="nin", impedance=50)

    def test_noise_input(self):
        inpt = Input(input_type="noise", node="nin", impedance="15.5k")
        self.assertEqual(inpt.input_type, "noise")
        self.assertEqual(inpt.impedance, 15.5e3)

    def test_noise_input_without_impedance(self):
        self.assertRaisesRegex(ValueError, r"impedance must be specified for noise input", Input,
                               input_type="noise", node="nin")
    
    def test_voltage_input(self):
        inpt = Input(input_type="voltage", node="nin")
        self.assertEqual(inpt.input_type, "voltage")
        # no input impedance
        self.assertEqual(inpt.impedance, None)

    def test_voltage_input_with_impedance(self):
        self.assertRaisesRegex(ValueError, r"impedance cannot be specified for non-noise input", Input,
                               input_type="voltage", node="nin", impedance="10.5M")

    def test_current_input(self):
        inpt = Input(input_type="current", node="nin")
        self.assertEqual(inpt.input_type, "current")
        # input impedance should be 1
        self.assertEqual(inpt.impedance, 1)

    def test_current_input_with_impedance(self):
        self.assertRaisesRegex(ValueError, r"impedance cannot be specified for non-noise input", Input,
                               input_type="current", node="nin", impedance="10.5M")