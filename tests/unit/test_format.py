"""Format tests"""

from unittest import TestCase

from zero.format import Quantity


class QuantityParserTestCase(TestCase):
    """Quantity parsing tests"""
    def test_invalid(self):
        # invalid characters
        for test_value in r" !\"€£$%^&\*\(\)\{\}\[\];:'@#~/\?><\\\|¬`":
            with self.subTest(msg="Test invalid quantity", quantity=test_value):
                self.assertRaisesRegex(ValueError, r"unrecognised quantity", Quantity, test_value)

        # invalid strings
        self.assertRaisesRegex(ValueError, r"unrecognised quantity", Quantity, "")
        self.assertRaisesRegex(ValueError, r"unrecognised quantity", Quantity, "invalid")

    def test_float_values(self):
        """Test parsing of float quantities"""
        self.assertAlmostEqual(Quantity(1.23), 1.23)
        self.assertAlmostEqual(Quantity(5.3e6), 5.3e6)
        self.assertAlmostEqual(Quantity(-765e3), -765e3)

    def test_string_values(self):
        """Test parsing of string quantities"""
        self.assertAlmostEqual(Quantity("1.23"), 1.23)
        self.assertAlmostEqual(Quantity("-765e3"), -765e3)
        self.assertAlmostEqual(Quantity("6.3e-2.3"), 6.3 * 10 ** -2.3)

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
        self.assertEqual(q.unit, None)
        q = Quantity("1.23 Hz")
        self.assertAlmostEqual(q, 1.23)
        self.assertEqual(q.unit, "Hz")
        q = Quantity("1.69pF")
        self.assertAlmostEqual(q, 1.69e-12)
        self.assertEqual(q.unit, "F")
        q = Quantity("3.21uH")
        self.assertAlmostEqual(q, 3.21e-6)
        self.assertEqual(q.unit, "H")
        q = Quantity("4.88MΩ")
        self.assertAlmostEqual(q, 4.88e6)
        self.assertEqual(q.unit, "Ω")

    def test_copy(self):
        """Test quantity copy constructor"""
        q = Quantity("1.23MHz")
        # objects (floats) equal
        self.assertEqual(q, Quantity(q))
        # floats equal
        self.assertEqual(float(q), float(Quantity(q)))
        # strings equal
        self.assertEqual(str(q), str(Quantity(q)))

class QuantityFormatterTestCase(TestCase):
    """Quantity formatting tests"""
    def test_default_format(self):
        """Test default quantities format"""
        # default precision is 4
        self.assertEqual(Quantity(1.23).format(), "1.2300")
        self.assertEqual(Quantity("4.56k").format(), "4.5600k")
        self.assertEqual(Quantity("7.89 M").format(), "7.8900M")
        self.assertEqual(Quantity("1.01 GHz").format(), "1.0100 GHz")

    def test_unit_format(self):
        """Test quantities with units format"""
        # SI scale and unit, default precision
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=True, show_si=True), "1.0100 GHz")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=True, show_si=True), "1.0100 nHz")

        # SI scale, but no unit, default precision
        self.assertEqual(Quantity("1.01 MHz").format(show_unit=False, show_si=True), "1.0100M")
        self.assertEqual(Quantity("1.01 uHz").format(show_unit=False, show_si=True), "1.0100µ")

        # unit, but no SI scale, default precision
        self.assertEqual(Quantity("1.01 kHz").format(show_unit=True, show_si=False), "1.0100e3 Hz")
        self.assertEqual(Quantity("1.01 mHz").format(show_unit=True, show_si=False), "1.0100e-3 Hz")

        # no unit nor SI scale, default precision
        self.assertEqual(Quantity("1.01 THz").format(show_unit=False, show_si=False), "1.0100e12")
        self.assertEqual(Quantity("1.01 pHz").format(show_unit=False, show_si=False), "1.0100e-12")

        # SI scale and unit, 0 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=True, show_si=True, precision=0), "1 GHz")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=True, show_si=True, precision=0), "1 nHz")

        # SI scale, but no unit, 1 decimal place
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=False, show_si=True, precision=1), "1.0G")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=False, show_si=True, precision=1), "1.0n")

        # unit, but no SI scale, 2 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=True, show_si=False, precision=2), "1.01e9 Hz")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=True, show_si=False, precision=2), "1.01e-9 Hz")

        # no unit nor SI scale, 3 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=False, show_si=False, precision=3), "1.010e9")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=False, show_si=False, precision=3), "1.010e-9")

        # with decimal place move
        self.assertEqual(Quantity("12345.01 GHz").format(show_unit=False, show_si=False, precision=3), "12.35e12")
        self.assertEqual(Quantity("12345.01 nHz").format(show_unit=False, show_si=False, precision=3), "12.35e-6")
        self.assertEqual(Quantity("0.0012345 nHz").format(show_unit=False, show_si=False, precision=3), "1.235e-12")

        # SI scale and unit, full precision
        self.assertEqual(Quantity("1.01 GHz").format(show_unit=True, show_si=True, precision="full"), "1.01 GHz")
        self.assertEqual(Quantity("1.01 nHz").format(show_unit=True, show_si=True, precision="full"), "1.01 nHz")

        # with decimal place move
        self.assertEqual(Quantity("12345.01 GHz").format(show_unit=True, show_si=True, precision="full"), "12.34501 THz")
        self.assertEqual(Quantity("12345.01 nHz").format(show_unit=True, show_si=True, precision="full"), "12.34501 µHz")

        # SI scale, but no unit, full precision
        self.assertEqual(Quantity("12.3456 GHz").format(show_unit=False, show_si=True, precision="full"), "12.3456G")
        self.assertEqual(Quantity("12.3456 nHz").format(show_unit=False, show_si=True, precision="full"), "12.3456n")

        # unit, but no SI scale, full precision
        self.assertEqual(Quantity("123.456789 GHz").format(show_unit=True, show_si=False, precision="full"), "123.456789e9 Hz")
        self.assertEqual(Quantity("123.456789 nHz").format(show_unit=True, show_si=False, precision="full"), "123.456789e-9 Hz")

        # with decimal place move
        self.assertEqual(Quantity("123456.789 GHz").format(show_unit=True, show_si=False, precision="full"), "123.456789e12 Hz")
        self.assertEqual(Quantity("123456.789 nHz").format(show_unit=True, show_si=False, precision="full"), "123.456789e-6 Hz")
        self.assertEqual(Quantity("0.00123456789 nHz").format(show_unit=True, show_si=False, precision="full"), "1.23456789e-12 Hz")

        # no unit nor SI scale, full precision
        self.assertEqual(Quantity("123.4567890123 GHz").format(show_unit=False, show_si=False, precision="full"), "123.4567890123e9")
        self.assertEqual(Quantity("123.4567890123 nHz").format(show_unit=False, show_si=False, precision="full"), "123.4567890123e-9")

        # with decimal place move
        self.assertEqual(Quantity("12345.67890123 GHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e12")
        self.assertEqual(Quantity("12345.67890123 nHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e-6")
        self.assertEqual(Quantity("0.001234567890123 nHz").format(show_unit=False, show_si=False, precision="full"), "1.234567890123e-12")

        # scales below f should default to exponential notation
        self.assertEqual(Quantity("0.001234567890123 fHz").format(show_unit=False, show_si=False, precision="full"), "1.234567890123e-18")
        self.assertEqual(Quantity("0.001234567890123 aHz").format(show_unit=False, show_si=False, precision="full"), "1.234567890123e-21")
        self.assertEqual(Quantity("0.001234567890123 zHz").format(show_unit=False, show_si=False, precision="full"), "1.234567890123e-24")
        self.assertEqual(Quantity("0.001234567890123 yHz").format(show_unit=False, show_si=False, precision="full"), "1.234567890123e-27")

        # scales above T should default to exponential notation
        self.assertEqual(Quantity("12345.67890123 THz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e15")
        self.assertEqual(Quantity("12345.67890123 PHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e18")
        self.assertEqual(Quantity("12345.67890123 EHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e21")
        self.assertEqual(Quantity("12345.67890123 ZHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e24")
        self.assertEqual(Quantity("12345.67890123 YHz").format(show_unit=False, show_si=False, precision="full"), "12.34567890123e27")
