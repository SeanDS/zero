"""AC noise analysis tests"""

from unittest import TestCase
import numpy as np

from zero import Circuit
from zero.analysis import AcNoiseAnalysis
from zero.components import Node


class AcNoiseAnalysisTestCase(TestCase):
    """AC noise analysis tests"""
    def setUp(self):
        self.circuit = Circuit()
        self.default_f = np.logspace(0, 5, 1000)
        self.default_element = Node("nin")

    @property
    def default_params(self):
        """Default analysis parameters"""
        return {"circuit": self.circuit, "element": self.default_element}

    def default_analysis(self):
        """Default analysis"""
        return AcNoiseAnalysis(**self.default_params)
