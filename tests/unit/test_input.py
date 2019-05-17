"""Input component tests"""

from unittest import TestCase

from zero.components import Input, Node


class InputTestCase(TestCase):
    """Test circuit input component"""
    def test_name(self):
        """Test input component name is 'input'"""
        inpt = Input(input_type="voltage", node="nin")
        self.assertEqual(inpt.name, "input")

    def test_name_cannot_be_set(self):
        """Test cannot set input component name"""
        self.assertRaises(TypeError, Input, name="abc", input_type="voltage", node="nin")

    def test_nodes(self):
        """Test input component nodes"""
        inpt = Input(input_type="voltage", node="nin")
        self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])

        inpt = Input(input_type="voltage", node_n="nin", node_p="gnd")
        self.assertEqual(inpt.nodes, [Node("nin"), Node("gnd")])

        inpt = Input(input_type="voltage", node_n="gnd", node_p="nin")
        self.assertEqual(inpt.nodes, [Node("gnd"), Node("nin")])

    def test_node_properties(self):
        """Test input component node properties"""
        inpt = Input(input_type="voltage", node_n="gnd", node_p="nin")
        self.assertEqual(inpt.node1, Node("gnd"))
        self.assertEqual(inpt.node2, Node("nin"))
        self.assertEqual(inpt.node_n, inpt.node1)
        self.assertEqual(inpt.node_p, inpt.node2)

    def test_invalid_node_combinations(self):
        """Test invalid input node combinations"""
        # node and node_p
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n",
                               Input, input_type="voltage", node="nin", node_p="nin")

        # node and node_n
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n",
                               Input, input_type="voltage", node="nin", node_n="nin")

        # node and node_n and node_p
        self.assertRaisesRegex(ValueError, r"node cannot be specified alongside node_p or node_n",
                               Input, input_type="voltage", node="nin", node_n="nin", node_p="nin")

        # node_n but not node_p
        self.assertRaisesRegex(ValueError, r"node_p and node_n must both be specified", Input,
                               input_type="voltage", node_n="nin")

        # node_p but not node_n
        self.assertRaisesRegex(ValueError, r"node_p and node_n must both be specified", Input,
                               input_type="voltage", node_p="nin")

    def test_noise_input(self):
        """Test noise input"""
        inpt = Input(input_type="voltage", node="nin", impedance="15.5k", is_noise=True)
        self.assertEqual(inpt.input_type, "voltage")
        self.assertEqual(inpt.impedance, 15.5e3)

    def test_noise_input_without_impedance(self):
        """Test noise input must have explicit impedance"""
        self.assertRaisesRegex(ValueError, r"impedance must be specified for noise input", Input,
                               input_type="voltage", node="nin", is_noise=True)

    def test_voltage_input(self):
        """Test voltage input"""
        inpt = Input(input_type="voltage", node="nin")
        self.assertEqual(inpt.input_type, "voltage")
        # no input impedance
        self.assertEqual(inpt.impedance, None)

    def test_voltage_input_with_impedance(self):
        """Test voltage input cannot have impedance"""
        self.assertRaisesRegex(ValueError, r"impedance cannot be specified for non-noise input",
                               Input, input_type="voltage", node="nin", impedance="10.5M")

    def test_current_input(self):
        """Test current input"""
        inpt = Input(input_type="current", node="nin")
        self.assertEqual(inpt.input_type, "current")
        # input impedance should be 1
        self.assertEqual(inpt.impedance, 1)

    def test_current_input_with_impedance(self):
        """Test current input cannot have impedance"""
        self.assertRaisesRegex(ValueError, r"impedance cannot be specified for non-noise input",
                               Input, input_type="current", node="nin", impedance="10.5M")
