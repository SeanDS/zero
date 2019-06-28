"""AC noise analysis integration tests"""

from unittest import TestCase
import numpy as np

from zero import Circuit
from zero.analysis import AcNoiseAnalysis


class AcNoiseAnalysisIntegrationTestCase(TestCase):
    """AC noise analysis tests"""
    def setUp(self):
        self.frequencies = np.logspace(0, 6, 1000)
        self.circuit = Circuit()
        self.circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
        self.circuit.add_resistor(value="430", node1="n1", node2="nm", name="r1")
        self.circuit.add_resistor(value="43k", node1="nm", node2="nout")
        self.circuit.add_capacitor(value="47p", node1="nm", node2="nout")
        self.circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")

    def test_input_noise_units(self):
        """Check units when projecting noise to input."""
        analysis = AcNoiseAnalysis(circuit=self.circuit)

        kwargs = {"frequencies": self.frequencies,
                  "node": "n1",
                  "sink": "nout",
                  "incoherent_sum": True}

        # Check the analysis without projecting.
        solution = analysis.calculate(input_type="current", **kwargs)
        self.assertRaises(ValueError, solution.get_noise, source="R(r1)", sink="input")
        rnoise = solution.get_noise(source="R(r1)", sink="nout")
        self.assertEqual(rnoise.sink_unit, "V")
        # Check the analysis, projecting to input, with voltage input.
        solution = analysis.calculate(input_type="voltage", input_refer=True, **kwargs)
        self.assertRaises(ValueError, solution.get_noise, source="R(r1)", sink="nout")
        rnoise = solution.get_noise(source="R(r1)", sink="n1")
        self.assertEqual(rnoise.sink_unit, "V")
        # Check the analysis, projecting to input, with current input.
        solution = analysis.calculate(input_type="current", input_refer=True, **kwargs)
        self.assertRaises(ValueError, solution.get_noise, source="R(r1)", sink="nout")
        rnoise = solution.get_noise(source="R(r1)", sink="input")
        self.assertEqual(rnoise.sink_unit, "A")
