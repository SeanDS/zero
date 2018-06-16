"""Circuit tests"""

from unittest import TestCase
import numpy as np

from circuit.analysis import AcSignalAnalysis
from circuit import Circuit
from circuit.components import Node

class AcSignalAnalysisTestCase(TestCase):
    def setUp(self):
        self.circuit = Circuit()
        self.default_f = np.logspace(0, 5, 1000)

    @property
    def default_params(self):
        """Default analysis parameters"""
        return {"circuit": self.circuit, "frequencies": self.default_f}

    def test_voltage_input(self):
        self.circuit.add_input(input_type="voltage", node="nin")
        analysis = AcSignalAnalysis(**self.default_params)

    def test_current_input(self):
        self.circuit.add_input(input_type="current", node="nin")
        analysis = AcSignalAnalysis(**self.default_params)

    def test_invalid_input(self):
        """When signal analysis made, input must be voltage or current"""
        self.circuit.add_input(input_type="noise", node="nin", impedance=10)
        self.assertRaisesRegex(ValueError, r"circuit input type must be either 'voltage' or 'current'",
                               AcSignalAnalysis, **self.default_params)