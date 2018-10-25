"""Mutual inductance integration tests"""

from unittest import TestCase
import numpy as np

from zero import Circuit
from zero.analysis import AcSignalAnalysis

class MutualInductanceTestCase(TestCase):
    """Mutual inductance tests"""
    def setUp(self):
        # triple transformer, winding ratio 1:3:3
        self.circuit = Circuit()
        self.circuit.add_input(input_type="voltage", node="n1")
        self.circuit.add_resistor(value="1k", node1="n1", node2="n2")
        self.circuit.add_inductor(value="1m", node1="n2", node2="gnd", name="l1")
        self.circuit.add_inductor(value="9m", node1="n3", node2="gnd", name="l2")
        self.circuit.add_inductor(value="9m", node1="n4", node2="gnd", name="l3")
        self.circuit.add_resistor(value="1k", node1="n3", node2="gnd")
        self.circuit.add_resistor(value="1k", node1="n4", node2="gnd")

    def test_matrix_impedance_elements(self):
        """Test impedance matrix elements are the same for both inductors"""
        # add mutual inductance between all three coils
        self.circuit.set_inductor_coupling("l1", "l2")
        self.circuit.set_inductor_coupling("l2", "l3")
        self.circuit.set_inductor_coupling("l1", "l3")

        # get circuit matrix, with prescaling switched off so we can directly compare values
        analysis = AcSignalAnalysis(circuit=self.circuit, prescale=False)

        # get inductors
        l1 = self.circuit["l1"]
        l2 = self.circuit["l2"]
        l3 = self.circuit["l3"]

        # get inductor current indices
        l1_index = analysis.component_matrix_index(l1)
        l2_index = analysis.component_matrix_index(l2)
        l3_index = analysis.component_matrix_index(l3)

        # check at various frequency decades
        for frequency in np.logspace(-3, 6, 10):
            with self.subTest(msg="Test impedance matrix elements for coupled inductors",
                              frequency=frequency):
                # get matrix for this frequency
                matrix = analysis.circuit_matrix(frequency)

                # there should be a term proportional to the current of each paired inductor on the
                # opposite inductor's equation
                self.assertEqual(matrix[l1_index, l2_index], l1.impedance_from(l2, frequency))
                self.assertEqual(matrix[l1_index, l2_index], matrix[l2_index, l1_index])
                self.assertEqual(matrix[l1_index, l3_index], l1.impedance_from(l3, frequency))
                self.assertEqual(matrix[l1_index, l3_index], matrix[l3_index, l1_index])
                self.assertEqual(matrix[l2_index, l3_index], l2.impedance_from(l3, frequency))
                self.assertEqual(matrix[l2_index, l3_index], matrix[l3_index, l2_index])
