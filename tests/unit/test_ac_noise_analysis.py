"""Circuit tests"""

from unittest import TestCase
import numpy as np

from circuit.analysis import AcNoiseAnalysis
from circuit import Circuit
from circuit.components import Node

class AcNoiseAnalysisTestCase(TestCase):
    def setUp(self):
        self.circuit = Circuit()
        self.default_f = np.logspace(0, 5, 1000)
        self.default_node = Node("nin")

    @property
    def default_params(self):
        """Default analysis parameters"""
        return {"circuit": self.circuit, "frequencies": self.default_f, "node": self.default_node}

    def default_analysis(self):
        return AcNoiseAnalysis(**self.default_params)

    def test_input(self):
        self.circuit.add_input(input_type="noise", node="nin", impedance=152.6)
        analysis = self.default_analysis()
        self.assertEqual(analysis.node, self.default_node)

    def test_invalid_input(self):
        """When noise analysis made, input type must be noise"""
        self.circuit.add_input(input_type="voltage", node="nin")
        self.assertRaisesRegex(ValueError, r"circuit input type must be 'noise'", self.default_analysis)