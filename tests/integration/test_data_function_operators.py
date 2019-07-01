"""Data function operator tests"""

from unittest import TestCase
import numpy as np

from zero import Circuit
from zero.data import Series, Response, NoiseDensity


class FunctionMathematicalOperationsTestCase(TestCase):
    """Function arithmetic tests."""
    def setUp(self):
        self.frequencies = np.logspace(0, 6, 1000)
        self.circuit = Circuit()
        self.circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
        self.circuit.add_resistor(value="430", node1="n1", node2="nm", name="r1")
        self.circuit.add_resistor(value="43k", node1="nm", node2="nout")
        self.circuit.add_capacitor(value="47p", node1="nm", node2="nout", name="op1")
        self.circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")

        vv_response_series = Series(self.frequencies, np.ones_like(self.frequencies) / 2)
        va_response_series = Series(self.frequencies, np.ones_like(self.frequencies) / 2)
        aa_response_series = Series(self.frequencies, np.ones_like(self.frequencies) / 2)
        av_response_series = Series(self.frequencies, np.ones_like(self.frequencies) / 2)
        noise_density_v = Series(self.frequencies, np.ones_like(self.frequencies) / 2)
        noise_density_a = Series(self.frequencies, np.ones_like(self.frequencies) / 2)

        self.response_vv = Response(source=self.circuit["nout"], sink=self.circuit["n1"],
                                    series=vv_response_series)
        self.response_va = Response(source=self.circuit["nout"], sink=self.circuit["r1"],
                                    series=va_response_series)
        self.response_aa = Response(source=self.circuit["r1"], sink=self.circuit["op1"],
                                    series=aa_response_series)
        self.response_av = Response(source=self.circuit["r1"], sink=self.circuit["n1"],
                                    series=av_response_series)
        self.noise_v = NoiseDensity(source=self.circuit["r1"].johnson_noise,
                                    sink=self.circuit["nout"], series=noise_density_v)
        self.noise_a = NoiseDensity(source=self.circuit["r1"].johnson_noise,
                                    sink=self.circuit["op1"], series=noise_density_a)

    def test_multiply_response_units(self):
        """Test units when multiplying response by response."""
        # Voltage-voltage response by voltage-voltage response.
        scaled_response = self.response_vv * self.response_vv
        self.assertEqual(scaled_response.sink_unit, "V")
        # Voltage-voltage response by voltage-current response.
        scaled_response = self.response_vv * self.response_va
        self.assertEqual(scaled_response.sink_unit, "A")
        # Voltage-current response by current-current response.
        scaled_response = self.response_va * self.response_aa
        self.assertEqual(scaled_response.sink_unit, "A")
        # Voltage-current response by current-voltage response.
        scaled_response = self.response_va * self.response_av
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-current response by current-current response.
        scaled_response = self.response_aa * self.response_aa
        self.assertEqual(scaled_response.sink_unit, "A")
        # Current-current response by current-voltage response.
        scaled_response = self.response_aa * self.response_av
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-voltage response by voltage-voltage response.
        scaled_response = self.response_av * self.response_vv
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-voltage response by voltage-current response.
        scaled_response = self.response_av * self.response_va
        self.assertEqual(scaled_response.sink_unit, "A")

        # Disallowed multiplications.
        self.assertRaises(ValueError, lambda: self.response_vv * self.response_av)
        self.assertRaises(ValueError, lambda: self.response_vv * self.response_aa)
        self.assertRaises(ValueError, lambda: self.response_va * self.response_vv)
        self.assertRaises(ValueError, lambda: self.response_va * self.response_va)
        self.assertRaises(ValueError, lambda: self.response_aa * self.response_vv)
        self.assertRaises(ValueError, lambda: self.response_aa * self.response_va)
        self.assertRaises(ValueError, lambda: self.response_av * self.response_av)
        self.assertRaises(ValueError, lambda: self.response_av * self.response_aa)

    def test_multiply_noise_by_response_units(self):
        """Test units when multiplying noise by response."""
        # Scale voltage noise by voltage-voltage response.
        scaled_noise = self.noise_v * self.response_vv
        self.assertEqual(scaled_noise.sink_unit, "V")
        # Scale voltage noise by voltage-current response.
        scaled_noise = self.noise_v * self.response_va
        self.assertEqual(scaled_noise.sink_unit, "A")
        # Scale current noise by current-current response.
        scaled_noise = self.noise_a * self.response_aa
        self.assertEqual(scaled_noise.sink_unit, "A")
        # Scale current noise by current-voltage response.
        scaled_noise = self.noise_a * self.response_av
        self.assertEqual(scaled_noise.sink_unit, "V")

    def test_multiply_noise_by_response_with_incompatible_source_invalid(self):
        """Test noise can't be multiplied by responses with source different to the noise sink."""
        self.assertRaises(ValueError, lambda: self.noise_v * self.response_av)
        self.assertRaises(ValueError, lambda: self.noise_v * self.response_aa)
        self.assertRaises(ValueError, lambda: self.noise_a * self.response_vv)
        self.assertRaises(ValueError, lambda: self.noise_a * self.response_va)

    def test_multiply_noise_by_noise_invalid(self):
        """Test that noise cannot be multiplied by noise."""
        self.assertRaises(ValueError, lambda: self.noise_v * self.noise_v)
        self.assertRaises(ValueError, lambda: self.noise_v * self.noise_a)
        self.assertRaises(ValueError, lambda: self.noise_a * self.noise_a)
        self.assertRaises(ValueError, lambda: self.noise_a * self.noise_v)

    def test_multiply_response_by_noise_invalid(self):
        """Test that a response cannot be multiplied by noise."""
        self.assertRaises(ValueError, lambda: self.response_vv * self.noise_v)
        self.assertRaises(ValueError, lambda: self.response_va * self.noise_a)
        self.assertRaises(ValueError, lambda: self.response_aa * self.noise_a)
        self.assertRaises(ValueError, lambda: self.response_av * self.noise_v)
