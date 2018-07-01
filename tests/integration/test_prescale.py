"""Prescale tests"""

from unittest import TestCase
import numpy as np

from circuit import Circuit
from circuit.analysis import AcSignalAnalysis, AcNoiseAnalysis

class PrescaleTestCase(TestCase):
    def setUp(self):
        self.f = np.logspace(0, 5, 300)

    def test_ac_signal_analysis(self):
        circuit = Circuit()
        circuit.add_input(input_type="voltage", node="n1")
        circuit.add_resistor(value="10k", node1="n1", node2="n2")
        circuit.add_capacitor(value="1n", node1="n1", node2="gnd")
        circuit.add_opamp(model="OP00", node1="n2", node2="gnd", node3="n3")
        circuit.add_resistor(value="100k", node1="n2", node2="n3")
        circuit.add_capacitor(value="1u", node1="n2", node2="n3")

        analysis_unscaled = AcSignalAnalysis(circuit=circuit, frequencies=self.f, prescale=True)
        analysis_scaled = AcSignalAnalysis(circuit=circuit, frequencies=self.f, prescale=False)

        analysis_unscaled.calculate()
        analysis_scaled.calculate()

        self.assertEqual(analysis_unscaled.solution, analysis_scaled.solution)

    def test_ac_noise_analysis(self):
        circuit = Circuit()
        circuit.add_input(input_type="noise", impedance="50", node="n1")
        circuit.add_resistor(value="10k", node1="n1", node2="n2")
        circuit.add_capacitor(value="1n", node1="n1", node2="gnd")
        circuit.add_opamp(model="OP00", node1="n2", node2="gnd", node3="n3")
        circuit.add_resistor(value="100k", node1="n2", node2="n3")
        circuit.add_capacitor(value="1u", node1="n2", node2="n3")

        noise_node = "n3"

        analysis_unscaled = AcNoiseAnalysis(circuit=circuit, node=noise_node, frequencies=self.f,
                                            prescale=True)
        analysis_scaled = AcNoiseAnalysis(circuit=circuit, node=noise_node, frequencies=self.f,
                                          prescale=False)

        analysis_unscaled.calculate()
        analysis_scaled.calculate()

        self.assertEqual(analysis_unscaled.solution, analysis_scaled.solution)
