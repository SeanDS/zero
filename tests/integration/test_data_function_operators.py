"""Data function operator tests"""

from ..data import ZeroDataTestCase


class FunctionMathematicalOperationsTestCase(ZeroDataTestCase):
    """Function mathematical operations tests."""
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
