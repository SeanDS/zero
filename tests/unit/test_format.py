"""Format tests"""

from unittest import TestCase

from zero.format import Quantity


class QuantityParserTestCase(TestCase):
    """Quantity parsing tests"""
    def test_float_values(self):
        """Test parsing of float quantities"""
        self.assertAlmostEqual(Quantity(1.23), 1.23)
        self.assertAlmostEqual(Quantity(5.3e6), 5.3e6)
        self.assertAlmostEqual(Quantity(-765e3), -765e3)

    def test_string_values(self):
        """Test parsing of string quantities"""
        self.assertAlmostEqual(Quantity("1.23"), 1.23)
        self.assertAlmostEqual(Quantity("-765e3"), -765e3)

    def test_string_values_with_si_scales(self):
        """Test parsing of string quantities with SI scales"""
        self.assertAlmostEqual(Quantity("1.23y"), 1.23e-24)
        self.assertAlmostEqual(Quantity("1.23z"), 1.23e-21)
        self.assertAlmostEqual(Quantity("1.23a"), 1.23e-18)
        self.assertAlmostEqual(Quantity("1.23f"), 1.23e-15)
        self.assertAlmostEqual(Quantity("1.23p"), 1.23e-12)
        self.assertAlmostEqual(Quantity("1.23n"), 1.23e-9)
        self.assertAlmostEqual(Quantity("1.23µ"), 1.23e-6)
        self.assertAlmostEqual(Quantity("1.23u"), 1.23e-6)
        self.assertAlmostEqual(Quantity("1.23m"), 1.23e-3)
        self.assertAlmostEqual(Quantity("1.23k"), 1.23e3)
        self.assertAlmostEqual(Quantity("1.23M"), 1.23e6)
        self.assertAlmostEqual(Quantity("1.23G"), 1.23e9)
        self.assertAlmostEqual(Quantity("1.23T"), 1.23e12)
        self.assertAlmostEqual(Quantity("1.23P"), 1.23e15)
        self.assertAlmostEqual(Quantity("1.23E"), 1.23e18)
        self.assertAlmostEqual(Quantity("1.23Z"), 1.23e21)
        self.assertAlmostEqual(Quantity("1.23Y"), 1.23e24)

    def test_string_values_with_units_and_si_scales(self):
        """Test parsing of string quantities with units and SI scales"""
        q = Quantity("1.23")
        self.assertAlmostEqual(q, 1.23)
        self.assertEqual(q.units, "")
        q = Quantity("1.23 Hz")
        self.assertAlmostEqual(q, 1.23)
        self.assertEqual(q.units, "Hz")
        q = Quantity("1.69pF")
        self.assertAlmostEqual(q, 1.69e-12)
        self.assertEqual(q.units, "F")
        q = Quantity("3.21uH")
        self.assertAlmostEqual(q, 3.21e-6)
        self.assertEqual(q.units, "H")
        q = Quantity("4.88MΩ")
        self.assertAlmostEqual(q, 4.88e6)
        self.assertEqual(q.units, "Ω")

    def test_copy(self):
        """Test quantity copy constructor"""
        q = Quantity("1.23MHz")
        # objects (floats) equal
        self.assertEqual(q, Quantity(q))
        # floats equal
        self.assertEqual(float(q), float(Quantity(q)))
        # strings equal
        self.assertEqual(str(q), str(Quantity(q)))
