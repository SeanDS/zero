"""AC signal analysis tests"""

from unittest import TestCase
import numpy as np

from zero.analysis import AcSignalAnalysis
from zero import Circuit


class AcSignalAnalysisTestCase(TestCase):
    """AC signal analysis tests"""
    def setUp(self):
        self.circuit = Circuit()
        self.default_f = np.logspace(0, 5, 1000)

    @property
    def default_params(self):
        """Default analysis parameters"""
        return {"circuit": self.circuit, "frequencies": self.default_f}

    def default_analysis(self):
        """Default analysis"""
        return AcSignalAnalysis(**self.default_params)

    def test_voltage_input(self):
        """Test set voltage input"""
        self.circuit.add_input(input_type="voltage", node="nin")
        self.default_analysis()

    def test_current_input(self):
        """Test set current input"""
        self.circuit.add_input(input_type="current", node="nin")
        self.default_analysis()
