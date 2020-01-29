"""AC noise analysis integration tests"""

from unittest import TestCase
import numpy as np

from zero import Circuit
from zero.analysis import AcNoiseAnalysis


class AcNoiseAnalysisIntegrationTestCase(TestCase):
    """AC noise analysis tests"""
    def setUp(self):
        self.f = np.logspace(0, 5, 100)

    def test_empty_circuit_calculation(self):
        """Test set voltage input"""
        circuit = Circuit()
        circuit.add_resistor(value="1k", node1="nin", node2="nout")
        analysis = AcNoiseAnalysis(circuit)
        for input_type in ("voltage", "current"):
            with self.subTest(input_type):
                analysis.calculate(frequencies=self.f, input_type=input_type, node="nin",
                                   sink="nout")
                self.assertEqual(analysis.n_freqs, len(self.f))
                self.assertCountEqual(analysis.element_names, ["input", "r1", "nin", "nout"])

    def test_input_noise_units(self):
        """Check units when projecting noise to input."""
        circuit = Circuit()
        circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
        circuit.add_resistor(value="430", node1="n1", node2="nm", name="r1")
        circuit.add_resistor(value="43k", node1="nm", node2="nout")
        circuit.add_capacitor(value="47p", node1="nm", node2="nout")
        circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")
        analysis = AcNoiseAnalysis(circuit=circuit)
        kwargs = {"frequencies": self.f, "node": "n1", "sink": "nout", "incoherent_sum": True}
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

    def test_changing_resistance_changes_noise(self):
        """Check that changing a resistance changes the corresponding Johnson noise."""
        circuit = Circuit()
        circuit.add_resistor(value="1k", node1="nin", node2="nout", name="r1")
        analysis = AcNoiseAnalysis(circuit=circuit)
        kwargs = {"frequencies": self.f, "node": "nin", "sink": "nout"}
        # First noise.
        solution1 = analysis.calculate(input_type="voltage", **kwargs)
        noise1 = solution1.get_noise(source="R(r1)", sink="nout")
        original_resistance = circuit["r1"].resistance
        for factor in (0.5, 2, 10, 1000):
            with self.subTest(factor):
                # Change resistance.
                circuit["r1"].resistance = original_resistance * factor
                # Second noise.
                solution2 = analysis.calculate(input_type="voltage", **kwargs)
                noise2 = solution2.get_noise(source="R(r1)", sink="nout")
                # Double the resistance gives only sqrt(2) more noise.
                self.assertTrue(np.allclose(noise1.spectral_density,
                                            1 / np.sqrt(factor) * noise2.spectral_density))
