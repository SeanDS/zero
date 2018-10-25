"""Data tests"""

from unittest import TestCase
import numpy as np

from zero.data import Series


class SeriesTestCase(TestCase):
    """Data series tests"""
    def setUp(self):
        # x-axis
        self.x = np.linspace(1, 10, 10)

        # randomly generated test data
        test_data = np.array([[0.87641139, 0.89256540],
                              [0.92233697, 0.14460983],
                              [0.46331225, 0.11945957],
                              [0.15163242, 0.31404200],
                              [0.07903830, 0.54334174],
                              [0.70267076, 0.40427563],
                              [0.77077928, 0.36697893],
                              [0.43051938, 0.08770115],
                              [0.71906046, 0.27778896],
                              [0.26178631, 0.02114534]])

        # first column is real, second imaginary
        self.data_cplx = test_data[:, 0] + 1j * test_data[:, 1]

        # separate real and imaginary parts
        self.data_re = test_data[:, 0]
        self.data_im = test_data[:, 1]

        # magnitude and phase equivalent
        self.data_mag_abs = np.abs(self.data_cplx)
        self.data_mag_db = 20 * np.log10(self.data_mag_abs)
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
        # real part cannot have imaginary element
        self.assertRaises(ValueError, Series.from_re_im, self.x, np.array([1, 2, 3+1j]),
                          np.array([1, 2, 3]))
        # imaginary part cannot have imaginary element
        self.assertRaises(ValueError, Series.from_re_im, self.x, np.array([1, 2, 3]),
                          np.array([1, 2, 3+1j]))

    def test_invalid_shape(self):
        """Test series constructor with invalid data shape"""
        self.assertRaises(ValueError, Series, x=np.array([1, 2, 3]), y=np.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, Series, x=np.array([[1, 2, 3], [4, 5, 6]]),
                          y=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
