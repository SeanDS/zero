"""Data function operator tests"""

from operator import mul, truediv, add, sub
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


class FunctionMathematicalOperationDataTestCase(ZeroDataTestCase):
    def setUp(self):
        self.scalars = [1, 1.5, -2.3, 1e2, -1e2, 6.7e-10, 3.78e10+4.802e9j, -6.8e-5-0.988e4j]
        self.scalars_inc_zero = [0] + self.scalars

    """Function mathematical operation data tests."""
    def _check_response_operate_scalar_right(self, response, scale, operation):
        scaled_response = operation(response, scale)
        self.assertTrue(np.allclose(scaled_response.complex_magnitude,
                                    operation(response.complex_magnitude, scale)))

    def _check_response_operate_scalar_left(self, response, scale, operation):
        scaled_response = operation(scale, response)
        self.assertTrue(np.allclose(scaled_response.complex_magnitude,
                                    operation(scale, response.complex_magnitude)))

    def _check_response_operate_scalar(self, response, scale, operation):
        self._check_response_operate_scalar_left(response, scale, operation)
        self._check_response_operate_scalar_right(response, scale, operation)

    def _check_response_multiply_scalar(self, response, scale):
        return self._check_response_operate_scalar(response, scale, mul)

    def _check_response_divide_scalar(self, response, scale):
        return self._check_response_operate_scalar(response, scale, truediv)

    def _check_response_multiply_response(self, left, right):
        scaled = left * right
        self.assertTrue(np.allclose(scaled.complex_magnitude,
                                    left.complex_magnitude * right.complex_magnitude))

    def _check_noise_multiply_response(self, noise, response):
        scaled_noise = noise * response
        self.assertTrue(np.allclose(scaled_noise.spectral_density,
                                    noise.spectral_density * response.magnitude))

    def _invalid_operation(self, operation, a, b):
        with self.subTest((operation, a, b)):
            self.assertRaises(TypeError, operation, a, b)

    def _invalid_operation_both_ways(self, operation, a, b):
        self._invalid_operation(operation, a, b)
        self._invalid_operation(operation, b, a)

    def test_valid_response_by_scalar_operations_both_ways(self):
        """Test data when multiplying response by scalar (and vice versa)."""
        f = self._freqs()
        for scalar in self.scalars_inc_zero:
            with self.subTest(scalar):
                self._check_response_multiply_scalar(self._v_v_response(f), scalar)
                self._check_response_multiply_scalar(self._v_i_response(f), scalar)
                self._check_response_multiply_scalar(self._i_i_response(f), scalar)
                self._check_response_multiply_scalar(self._i_v_response(f), scalar)

    def test_invalid_response_by_scalar_operations_both_ways(self):
        """Test that certain operators between a response and a scalar are invalid."""
        f = self._freqs()
        for scalar in self.scalars_inc_zero:
            for operation in (add, sub):
                self._invalid_operation_both_ways(operation, self._v_v_response(f), scalar)
                self._invalid_operation_both_ways(operation, self._v_i_response(f), scalar)
                self._invalid_operation_both_ways(operation, self._i_i_response(f), scalar)
                self._invalid_operation_both_ways(operation, self._i_v_response(f), scalar)

    def test_valid_divide_response_by_scalar_both_ways(self):
        """Test data when dividing response by scalar (and vice versa)."""
        f = self._freqs()
        for scalar in self.scalars:
            with self.subTest(scalar):
                self._check_response_divide_scalar(self._v_v_response(f), scalar)
                self._check_response_divide_scalar(self._v_i_response(f), scalar)
                self._check_response_divide_scalar(self._i_i_response(f), scalar)
                self._check_response_divide_scalar(self._i_v_response(f), scalar)

    def test_valid_multiply_response_by_response(self):
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

    def test_invalid_response_by_response_operations(self):
        """Test that certain operators between two responses are invalid."""
        f = self._freqs()
        for operation in (truediv, add, sub):
            self._invalid_operation(operation, self._v_v_response(f), self._v_v_response(f))
            self._invalid_operation(operation, self._v_v_response(f), self._v_i_response(f))
            self._invalid_operation(operation, self._v_i_response(f), self._i_i_response(f))
            self._invalid_operation(operation, self._v_i_response(f), self._i_v_response(f))
            self._invalid_operation(operation, self._i_i_response(f), self._i_i_response(f))
            self._invalid_operation(operation, self._i_i_response(f), self._i_v_response(f))
            self._invalid_operation(operation, self._i_v_response(f), self._v_v_response(f))
            self._invalid_operation(operation, self._i_v_response(f), self._v_i_response(f))

    def test_invalid_response_by_noise_operations(self):
        """Test that certain operators between a response and a noise are invalid."""
        f = self._freqs()
        for operation in (mul, truediv):
            self._invalid_operation(operation, self._v_v_response(f), self._vnoise_at_node(f))
            self._invalid_operation(operation, self._v_i_response(f), self._vnoise_at_comp(f))
            self._invalid_operation(operation, self._i_i_response(f), self._vnoise_at_comp(f))
            self._invalid_operation(operation, self._i_v_response(f), self._vnoise_at_node(f))

    def test_invalid_noise_by_noise_operations(self):
        """Test that certain operators between two noise densities are invalid."""
        f = self._freqs()
        for operation in (mul, truediv):
            self._invalid_operation(operation, self._vnoise_at_node(f), self._vnoise_at_node(f))
            self._invalid_operation(operation, self._vnoise_at_node(f), self._vnoise_at_comp(f))
            self._invalid_operation(operation, self._vnoise_at_comp(f), self._vnoise_at_comp(f))
            self._invalid_operation(operation, self._vnoise_at_comp(f), self._vnoise_at_node(f))

    def test_valid_multiply_noise_by_response(self):
        """Test data when multiplying noise by response."""
        f = self._freqs()
        self._check_noise_multiply_response(self._vnoise_at_node(f), self._v_v_response(f))
        self._check_noise_multiply_response(self._vnoise_at_node(f), self._v_i_response(f))
        self._check_noise_multiply_response(self._vnoise_at_comp(f), self._i_i_response(f))
        self._check_noise_multiply_response(self._vnoise_at_comp(f), self._i_v_response(f))
        self._check_noise_multiply_response(self._inoise_at_node(f), self._v_v_response(f))
        self._check_noise_multiply_response(self._inoise_at_node(f), self._v_i_response(f))
        self._check_noise_multiply_response(self._inoise_at_comp(f), self._i_i_response(f))
        self._check_noise_multiply_response(self._inoise_at_comp(f), self._i_v_response(f))

    def test_invalid_noise_by_response_operations(self):
        """Test that certain operators between a noise density and a response are invalid."""
        f = self._freqs()
        for operation in (truediv, add, sub):
            self._invalid_operation(operation, self._vnoise_at_node(f), self._v_v_response(f))
            self._invalid_operation(operation, self._vnoise_at_node(f), self._v_i_response(f))
            self._invalid_operation(operation, self._vnoise_at_comp(f), self._i_i_response(f))
            self._invalid_operation(operation, self._vnoise_at_comp(f), self._i_v_response(f))
