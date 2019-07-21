"""Data tests"""

import numpy as np
from zero.data import Series, MultiNoiseDensity
from zero.misc import mag_to_db
from ..data import ZeroDataTestCase

class SeriesTestCase(ZeroDataTestCase):
    """Data series tests"""
    def setUp(self):
        # X-axis.
        self.x = self._freqs(10)
        # Randomly generated test data.
        test_data = self._data((len(self.x), 2))
        # First column is real, second imaginary.
        self.data_cplx = test_data[:, 0] + 1j * test_data[:, 1]
        # Separate real and imaginary parts.
        self.data_re = test_data[:, 0]
        self.data_im = test_data[:, 1]
        # Magnitude and phase equivalent.
        self.data_mag_abs = np.abs(self.data_cplx)
        self.data_mag_db = mag_to_db(self.data_mag_abs)
        self.data_phase_rad = np.angle(self.data_cplx)
        self.data_phase_deg = np.degrees(self.data_phase_rad)

    def test_from_mag_phase_default(self):
        """Test magnitude/phase series factory using default scale"""
        self.assertEqual(Series.from_mag_phase(self.x, self.data_mag_abs, self.data_phase_deg),
                         Series.from_mag_phase(self.x, self.data_mag_abs, self.data_phase_deg,
                                               mag_scale="abs", phase_scale="deg"))

    def test_from_mag_phase_abs_deg(self):
        """Test magnitude/phase series factory using abs/deg scale"""
        self.assertEqual(Series(self.x, self.data_cplx),
                         Series.from_mag_phase(self.x, self.data_mag_abs, self.data_phase_deg,
                                               mag_scale="abs", phase_scale="deg"))

    def test_from_mag_phase_abs_rad(self):
        """Test magnitude/phase series factory using abs/rad scale"""
        self.assertEqual(Series(self.x, self.data_cplx),
                         Series.from_mag_phase(self.x, self.data_mag_abs, self.data_phase_rad,
                                               mag_scale="abs", phase_scale="rad"))

    def test_from_mag_phase_db_deg(self):
        """Test magnitude/phase series factory using db/deg scale"""
        self.assertEqual(Series(self.x, self.data_cplx),
                         Series.from_mag_phase(self.x, self.data_mag_db, self.data_phase_rad,
                                               mag_scale="db", phase_scale="rad"))

    def test_from_mag_phase_db_rad(self):
        """Test magnitude/phase series factory using db/rad scale"""
        self.assertEqual(Series(self.x, self.data_cplx),
                         Series.from_mag_phase(self.x, self.data_mag_db, self.data_phase_deg,
                                               mag_scale="db", phase_scale="deg"))

    def test_from_mag_phase_invalid_scale(self):
        """Test magnitude/phase series factory invalid scales"""
        # "ab" and "minute" invalid
        self.assertRaises(ValueError, Series.from_mag_phase, self.x, self.data_mag_db,
                          self.data_phase_rad, mag_scale="ab", phase_scale="rad")
        self.assertRaises(ValueError, Series.from_mag_phase, self.x, self.data_mag_db,
                          self.data_phase_rad, mag_scale="db", phase_scale="minute")

    def test_from_re_im(self):
        """Test real/imaginary series factory using default scale"""
        self.assertEqual(Series(self.x, self.data_cplx),
                         Series.from_re_im(self.x, re=self.data_re, im=self.data_im))

    def test_from_re_im_invalid_parts(self):
        """Test real/imaginary series factory invalid scales"""
        # Real part cannot have imaginary element.
        self.assertRaises(ValueError, Series.from_re_im, self.x, np.array([1, 2, 3+1j]),
                          np.array([1, 2, 3]))
        # Imaginary part cannot have imaginary element.
        self.assertRaises(ValueError, Series.from_re_im, self.x, np.array([1, 2, 3]),
                          np.array([1, 2, 3+1j]))

    def test_invalid_shape(self):
        """Test series constructor with invalid data shape"""
        self.assertRaises(ValueError, Series, x=np.array([1, 2, 3]), y=np.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, Series, x=np.array([[1, 2, 3], [4, 5, 6]]),
                          y=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    def test_addition(self):
        """Test series addition."""
        series1 = Series(self.x, self.data_cplx)
        series2 = Series(self.x, self.data_cplx)
        series3 = Series(self.x, self.data_cplx)
        combined = series1 + series2 + series3
        self.assertTrue(np.allclose(combined.x, combined.x))
        self.assertTrue(np.allclose(combined.y, self.data_cplx * 3))

    def test_addition_scalar(self):
        """Test series scalar addition."""
        series = Series(self.x, self.data_cplx)
        scaled = series + 5
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx + 5))

    def test_subtract(self):
        """Test series subtraction."""
        series1 = Series(self.x, self.data_cplx)
        series2 = Series(self.x, self.data_cplx)
        series3 = Series(self.x, self.data_cplx)
        combined = series1 - series2 - series3
        self.assertTrue(np.allclose(combined.x, combined.x))
        self.assertTrue(np.allclose(combined.y, self.data_cplx * -1))

    def test_subtract_scalar(self):
        """Test series scalar subtraction."""
        series = Series(self.x, self.data_cplx)
        scaled = series - 5
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx - 5))

    def test_multiply(self):
        """Test series multiplication."""
        series1 = Series(self.x, self.data_cplx)
        series2 = Series(self.x, self.data_cplx)
        series3 = Series(self.x, self.data_cplx)
        combined = series1 * series2 * series3
        self.assertTrue(np.allclose(combined.x, series1.x))
        self.assertTrue(np.allclose(combined.y, self.data_cplx ** 3))

    def test_multiply_self(self):
        """Test series multiplication with self."""
        series = Series(self.x, self.data_cplx)
        combined = series * series * series
        # Multiplication should return a new object, so there shouldn't be issues with data
        # changing later.
        series.y = np.zeros_like(series.y)
        self.assertTrue(np.allclose(combined.x, series.x))
        self.assertTrue(np.allclose(combined.y, self.data_cplx ** 3))

    def test_multiply_scalar(self):
        """Test series scalar multiplication."""
        series = Series(self.x, self.data_cplx)
        # Right multiplication.
        scaled = series * 5
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx * 5))
        # Left multiplication.
        scaled = 5 * series
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx * 5))

    def test_divide(self):
        """Test series division."""
        series1 = Series(self.x, self.data_cplx)
        series2 = Series(self.x, self.data_cplx)
        combined = series1 / series2
        self.assertTrue(np.allclose(combined.x, series1.x))
        self.assertTrue(np.allclose(combined.y, np.ones_like(self.data_cplx)))

    def test_divide_self(self):
        """Test series division with self."""
        series = Series(self.x, self.data_cplx)
        combined = series / series
        # Division should return a new object, so there shouldn't be issues with data changing
        # later.
        series.y = np.zeros_like(series.y)
        self.assertTrue(np.allclose(combined.x, series.x))
        self.assertTrue(np.allclose(combined.y, np.ones_like(self.data_cplx)))

    def test_divide_scalar(self):
        """Test series scalar division."""
        series = Series(self.x, self.data_cplx)
        # Right division.
        scaled = series / 5
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx / 5))
        # Left (reflexive) division.
        scaled = 5 / series
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, 5 / self.data_cplx))

    def test_exponentiate(self):
        """Test series exponentiation."""
        series1 = Series(self.x, self.data_cplx)
        series2 = Series(self.x, self.data_cplx)
        series3 = Series(self.x, self.data_cplx)
        combined = series1 ** series2 ** series3
        self.assertTrue(np.allclose(combined.x, combined.x))
        self.assertTrue(np.allclose(combined.y, self.data_cplx ** self.data_cplx ** self.data_cplx))

    def test_exponentiate_scalar(self):
        """Test series scalar exponentiation."""
        series = Series(self.x, self.data_cplx)
        scaled = series ** 5
        self.assertTrue(np.allclose(scaled.x, series.x))
        self.assertTrue(np.allclose(scaled.y, self.data_cplx ** 5))

    def test_negate(self):
        """Test series negation."""
        series = Series(self.x, self.data_cplx)
        negated = -series
        # Negation should return a new object, so there shouldn't be issues with data changing
        # later.
        series.y = np.zeros_like(series.y)
        self.assertTrue(np.allclose(negated.x, series.x))
        self.assertTrue(np.allclose(negated.y, -self.data_cplx))

    def test_inverse(self):
        """Test series inversion."""
        series = Series(self.x, self.data_cplx)
        # Standard inverse.
        inverted = series.inverse()
        self.assertTrue(np.allclose(inverted.x, series.x))
        self.assertTrue(np.allclose(inverted.y, 1 / self.data_cplx))
        # Alternate inverse.
        inverted = 1 / series
        self.assertTrue(np.allclose(inverted.x, series.x))
        self.assertTrue(np.allclose(inverted.y, 1 / self.data_cplx))

    def test_inverse_self(self):
        """Test series inversion with self."""
        series = Series(self.x, self.data_cplx)
        inverted = series.inverse()
        # Division should return a new object, so there shouldn't be issues with data changing
        # later.
        series.y = np.zeros_like(series.y)
        self.assertTrue(np.allclose(inverted.x, series.x))
        self.assertTrue(np.allclose(inverted.y, 1 / self.data_cplx))


class MultiNoiseDensityTestCase(ZeroDataTestCase):
    """MultiNoiseDensity tests."""
    def test_constituent_noise_sum_equal_total_noise_sum(self):
        f = self._freqs()
        sink = self._resistor()
        noise1 = self._vnoise_at_comp(f, sink=sink)
        noise2 = self._vnoise_at_comp(f, sink=sink) # Share sink.
        constituents = [noise1, noise2]
        sum_data = np.sqrt(sum([noise.spectral_density ** 2 for noise in constituents]))
        sum_series = self._series(f, sum_data)
        noisesum1 = MultiNoiseDensity(sink=sink, constituents=constituents)
        noisesum2 = MultiNoiseDensity(sources=[noise1.source, noise2.source],
                                      sink=sink, series=sum_series)
        self.assertTrue(noisesum1.equivalent(noisesum2))
