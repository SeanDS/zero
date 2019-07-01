"""Data function operator tests"""

import numpy as np
from ..data import ZeroDataTestCase


class FunctionMathematicalOperationUnitsTestCase(ZeroDataTestCase):
    """Function mathematical operation units tests."""
    def test_multiply_response_units(self):
        """Test units when multiplying response by response."""
        f = self._freqs()
        # Voltage-voltage response by voltage-voltage response.
        scaled_response = self._v_v_response(f) * self._v_v_response(f)
        self.assertEqual(scaled_response.sink_unit, "V")
        # Voltage-voltage response by voltage-current response.
        scaled_response = self._v_v_response(f) * self._v_i_response(f)
        self.assertEqual(scaled_response.sink_unit, "A")
        # Voltage-current response by current-current response.
        scaled_response = self._v_i_response(f) * self._i_i_response(f)
        self.assertEqual(scaled_response.sink_unit, "A")
        # Voltage-current response by current-voltage response.
        scaled_response = self._v_i_response(f) * self._i_v_response(f)
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-current response by current-current response.
        scaled_response = self._i_i_response(f) * self._i_i_response(f)
        self.assertEqual(scaled_response.sink_unit, "A")
        # Current-current response by current-voltage response.
        scaled_response = self._i_i_response(f) * self._i_v_response(f)
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-voltage response by voltage-voltage response.
        scaled_response = self._i_v_response(f) * self._v_v_response(f)
        self.assertEqual(scaled_response.sink_unit, "V")
        # Current-voltage response by voltage-current response.
        scaled_response = self._i_v_response(f) * self._v_i_response(f)
        self.assertEqual(scaled_response.sink_unit, "A")

        # Disallowed multiplications.
        self.assertRaises(ValueError, lambda: self._v_v_response(f) * self._i_v_response(f))
        self.assertRaises(ValueError, lambda: self._v_v_response(f) * self._i_i_response(f))
        self.assertRaises(ValueError, lambda: self._v_i_response(f) * self._v_v_response(f))
        self.assertRaises(ValueError, lambda: self._v_i_response(f) * self._v_i_response(f))
        self.assertRaises(ValueError, lambda: self._i_i_response(f) * self._v_v_response(f))
        self.assertRaises(ValueError, lambda: self._i_i_response(f) * self._v_i_response(f))
        self.assertRaises(ValueError, lambda: self._i_v_response(f) * self._i_v_response(f))
        self.assertRaises(ValueError, lambda: self._i_v_response(f) * self._i_i_response(f))

    def test_multiply_noise_by_response_units(self):
        """Test units when multiplying noise by response."""
        f = self._freqs()
        # Scale voltage noise by voltage-voltage response.
        scaled_noise = self._vnoise_at_node(f) * self._v_v_response(f)
        self.assertEqual(scaled_noise.sink_unit, "V")
        # Scale voltage noise by voltage-current response.
        scaled_noise = self._vnoise_at_node(f) * self._v_i_response(f)
        self.assertEqual(scaled_noise.sink_unit, "A")
        # Scale current noise by current-current response.
        scaled_noise = self._vnoise_at_comp(f) * self._i_i_response(f)
        self.assertEqual(scaled_noise.sink_unit, "A")
        # Scale current noise by current-voltage response.
        scaled_noise = self._vnoise_at_comp(f) * self._i_v_response(f)
        self.assertEqual(scaled_noise.sink_unit, "V")

    def test_multiply_noise_by_response_with_incompatible_source_invalid(self):
        """Test noise can't be multiplied by responses with source different to the noise sink."""
        f = self._freqs()
        self.assertRaises(ValueError, lambda: self._vnoise_at_node(f) * self._i_v_response(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_node(f) * self._i_i_response(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_comp(f) * self._v_v_response(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_comp(f) * self._v_i_response(f))

    def test_multiply_noise_by_noise_invalid(self):
        """Test that noise cannot be multiplied by noise."""
        f = self._freqs()
        self.assertRaises(ValueError, lambda: self._vnoise_at_node(f) * self._vnoise_at_node(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_node(f) * self._vnoise_at_comp(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_comp(f) * self._vnoise_at_comp(f))
        self.assertRaises(ValueError, lambda: self._vnoise_at_comp(f) * self._vnoise_at_node(f))

    def test_multiply_response_by_noise_invalid(self):
        """Test that a response cannot be multiplied by noise."""
        f = self._freqs()
        self.assertRaises(ValueError, lambda: self._v_v_response(f) * self._vnoise_at_node(f))
        self.assertRaises(ValueError, lambda: self._v_i_response(f) * self._vnoise_at_comp(f))
        self.assertRaises(ValueError, lambda: self._i_i_response(f) * self._vnoise_at_comp(f))
        self.assertRaises(ValueError, lambda: self._i_v_response(f) * self._vnoise_at_node(f))


class FunctionMathematicalOperationDataTestCase(ZeroDataTestCase):
    """Function mathematical operation data tests."""
    def _check_response_multiply_scalar_right(self, response, scale):
        scaled_response = response * scale
        self.assertTrue(np.allclose(scaled_response.complex_magnitude,
                                    response.complex_magnitude * scale))

    def _check_response_multiply_scalar_left(self, response, scale):
        scaled_response = scale * response
        self.assertTrue(np.allclose(scaled_response.complex_magnitude,
                                    scale * response.complex_magnitude))

    def _check_response_multiply_scalar(self, response, scale):
        self._check_response_multiply_scalar_left(response, scale)
        self._check_response_multiply_scalar_right(response, scale)

    def _check_response_multiply_response(self, left, right):
        scaled = left * right
        self.assertTrue(np.allclose(scaled.complex_magnitude,
                                    left.complex_magnitude * right.complex_magnitude))

    def test_multiply_response_by_scalar(self):
        """Test data when multiplying response by scalar."""
        f = self._freqs()
        for scalar in (0, 1, 1.5, -2.3, 1e2, -1e2, 6.7e-10, 3.78e10+4.802e9j):
            self._check_response_multiply_scalar(self._v_v_response(f), scalar)
            self._check_response_multiply_scalar(self._v_i_response(f), scalar)
            self._check_response_multiply_scalar(self._i_i_response(f), scalar)
            self._check_response_multiply_scalar(self._i_v_response(f), scalar)

    def test_multiply_response_by_response(self):
        """Test data when multiplying response by response."""
        f = self._freqs()
        self._check_response_multiply_response(self._v_v_response(f), self._v_v_response(f))
        self._check_response_multiply_response(self._v_v_response(f), self._v_i_response(f))
        self._check_response_multiply_response(self._v_i_response(f), self._i_i_response(f))
        self._check_response_multiply_response(self._v_i_response(f), self._i_v_response(f))
        self._check_response_multiply_response(self._i_i_response(f), self._i_i_response(f))
        self._check_response_multiply_response(self._i_i_response(f), self._i_v_response(f))
        self._check_response_multiply_response(self._i_v_response(f), self._v_v_response(f))
        self._check_response_multiply_response(self._i_v_response(f), self._v_i_response(f))
