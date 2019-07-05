"""AC signal analysis integration tests"""

from unittest import TestCase
import numpy as np

from zero.analysis import AcSignalAnalysis
from zero import Circuit


class AcSignalAnalysisTestCase(TestCase):
    """AC signal analysis tests"""
    def setUp(self):
        self.f = np.logspace(0, 5, 1000)

    def test_empty_circuit_calculation(self):
        """Test set voltage input"""
        circuit = Circuit()
        analysis = AcSignalAnalysis(circuit)
        for input_type in ("voltage", "current"):
            with self.subTest(input_type):
                analysis.calculate(frequencies=self.f, input_type=input_type, node="nin")
                self.assertEqual(analysis.n_freqs, len(self.f))
                # Circuit should have input component and node.
                self.assertCountEqual(analysis.element_names, ["input", "nin"])
