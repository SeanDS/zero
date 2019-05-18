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
        return {"circuit": self.circuit}

    def default_analysis(self):
        """Default analysis"""
        return AcSignalAnalysis(**self.default_params)
