"""Mutual inductance integration tests"""

from unittest import TestCase
from copy import copy
import numpy as np

from zero import Circuit
from zero.analysis import AcSignalAnalysis


class MockMutualInductanceAcSignalAnalysis(AcSignalAnalysis):
    """AC signal analysis where the circuit matrix can be retrieved directly."""
    def set_input(self, **kwargs):
        # Make a copy of the circuit and set its input. This is normally performed by
        # _do_calculate().
        self._current_circuit = copy(self.circuit)
        self._set_input(**kwargs)

    def circuit_matrix(self, frequency):
        return super().circuit_matrix(frequency)

    def get_component_from_current_circuit(self, name):
        return self._current_circuit[name]


class MutualInductanceTestCase(TestCase):
    """Mutual inductance tests"""
    def setUp(self):
        # Triple transformer, winding ratio 1:3:3.
        self.circuit = Circuit()
        self.circuit.add_resistor(value="1k", node1="n1", node2="n2")
        self.circuit.add_inductor(value="1m", node1="n2", node2="gnd", name="l1")
        self.circuit.add_inductor(value="9m", node1="n3", node2="gnd", name="l2")
        self.circuit.add_inductor(value="9m", node1="n4", node2="gnd", name="l3")
        self.circuit.add_resistor(value="1k", node1="n3", node2="gnd")
        self.circuit.add_resistor(value="1k", node1="n4", node2="gnd")

    def test_matrix_impedance_elements(self):
        """Test impedance matrix elements are the same for both inductors"""
        # Add mutual inductance between all three coils.
        self.circuit.set_inductor_coupling("l1", "l2")
        self.circuit.set_inductor_coupling("l2", "l3")
        self.circuit.set_inductor_coupling("l1", "l3")

        # Get circuit matrix, with prescaling switched off so we can directly compare values.
        analysis = MockMutualInductanceAcSignalAnalysis(circuit=self.circuit)

        # Add circuit input.
        analysis.set_input(input_type="voltage", node="n1")

        # Get inductors.
        l1 = analysis.get_component_from_current_circuit("l1")
        l2 = analysis.get_component_from_current_circuit("l2")
        l3 = analysis.get_component_from_current_circuit("l3")

        # Get inductor current indices.
        l1_index = analysis.component_matrix_index(l1)
        l2_index = analysis.component_matrix_index(l2)
        l3_index = analysis.component_matrix_index(l3)

        # Check at various frequency decades.
        for frequency in np.logspace(-3, 6, 10):
            with self.subTest(msg="Test impedance matrix elements for coupled inductors",
                              frequency=frequency):
                # Get matrix for this frequency.
                matrix = analysis.circuit_matrix(frequency)

                # There should be a term proportional to the current of each paired inductor on the
                # opposite inductor's equation.
                self.assertEqual(matrix[l1_index, l2_index], l1.impedance_from(l2, frequency))
                self.assertEqual(matrix[l1_index, l2_index], matrix[l2_index, l1_index])
                self.assertEqual(matrix[l1_index, l3_index], l1.impedance_from(l3, frequency))
                self.assertEqual(matrix[l1_index, l3_index], matrix[l3_index, l1_index])
                self.assertEqual(matrix[l2_index, l3_index], l2.impedance_from(l3, frequency))
                self.assertEqual(matrix[l2_index, l3_index], matrix[l3_index, l2_index])
