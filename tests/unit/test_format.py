"""Format tests"""

from unittest import TestCase

from circuit.format import Quantity

class QuantityInitTestCase(TestCase):
    def setUp(self):
        pass

    def test_float(self):
        self.assertEqual(Quantity(1.23), 1.23)
        self.assertEqual(Quantity(5.3e6), 5.3e6)
        self.assertEqual(Quantity(-765e3), -765e3)
    
    def test_str(self):
        self.assertEqual(Quantity("1.23"), 1.23)
        self.assertEqual(Quantity("-765e3"), -765e3)
        self.assertEqual(Quantity("6.3e-2.3"), 6.3 * 10 ** -2.3)
        
        # SI scales
        self.assertEqual(Quantity("1.23y"), 1.23e-24)
        self.assertEqual(Quantity("1.23z"), 1.23e-21)
        self.assertEqual(Quantity("1.23a"), 1.23e-18)
        self.assertEqual(Quantity("1.23f"), 1.23e-15)
        self.assertEqual(Quantity("1.23p"), 1.23e-12)
        self.assertEqual(Quantity("1.23n"), 1.23e-9)
        self.assertEqual(Quantity("1.23µ"), 1.23e-6)
        self.assertEqual(Quantity("1.23u"), 1.23e-6)
        self.assertEqual(Quantity("1.23m"), 1.23e-3)
        self.assertEqual(Quantity("1.23k"), 1.23e3)
        self.assertEqual(Quantity("1.23M"), 1.23e6)
        self.assertEqual(Quantity("1.23G"), 1.23e9)
        self.assertEqual(Quantity("1.23T"), 1.23e12)
        self.assertEqual(Quantity("1.23E"), 1.23e15)
        self.assertEqual(Quantity("1.23Z"), 1.23e18)
        self.assertEqual(Quantity("1.23Y"), 1.23e21)

        # units
        q = Quantity("1.23")
        self.assertEqual(q, 1.23)
        self.assertEqual(q.unit, None)
        q = Quantity("1.23 Hz")
        self.assertEqual(q, 1.23)
        self.assertEqual(q.unit, "Hz")
        q = Quantity("1.69pF")
        self.assertEqual(q, 1.69e-12)
        self.assertEqual(q.unit, "F")
        q = Quantity("3.21uH")
        self.assertEqual(q, 3.21e-6)
        self.assertEqual(q.unit, "H")
        q = Quantity("4.88MΩ")
        self.assertEqual(q, 4.88e6)
        self.assertEqual(q.unit, "Ω")
    
    def test_copy(self):
        """Copy constructor"""
        q = Quantity("1.23MHz")
        # objects (floats) equal
        self.assertEqual(q, Quantity(q))
        # floats equal
        self.assertEqual(float(q), float(Quantity(q)))
        # strings equal
        self.assertEqual(str(q), str(Quantity(q)))

class QuantityFormatTestCase(TestCase):
    def setUp(self):
        pass

    def test_default_format(self):
        self.assertEqual(Quantity(1.23).format(), "1.23")
        self.assertEqual(Quantity("4.56k").format(), "4.56 k")
        self.assertEqual(Quantity("7.89 M").format(), "7.89 M")
        self.assertEqual(Quantity("1.01 GHz").format(), "1.01 GHz")
    
    def test_unit_format(self):
        # SI scale and unit, full precision
        self.assertEqual(Quantity("1.01 GHz").format(unit=True, si=True), "1.01 GHz")
        # SI scale, but no unit, full precision
        self.assertEqual(Quantity("1.01 GHz").format(unit=False, si=True), "1.01 G")
        # unit, but no SI scale, full precision
        self.assertEqual(Quantity("1.01 GHz").format(unit=True, si=False), "1010000000.0 Hz")
        # no unit nor SI scale, full precision
        self.assertEqual(Quantity("1.01 GHz").format(unit=False, si=False), "1010000000.0")
    
        # SI scale and unit, 0 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(unit=True, si=True, precision=0), "1 GHz")
        # SI scale, but no unit, 1 decimal place
        self.assertEqual(Quantity("1.01 GHz").format(unit=False, si=True, precision=1), "1.0 G")
        # unit, but no SI scale, 2 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(unit=True, si=False, precision=2), "1010000000.00 Hz")
        # no unit nor SI scale, 3 decimal places
        self.assertEqual(Quantity("1.01 GHz").format(unit=False, si=False, precision=3), "1010000000.000")