"""Configuration component parser tests"""

from unittest import TestCase

from zero.config import LibraryOpAmp


class LibraryOpAmpTestCase(TestCase):
    """Library op-amp parser tests"""
    def test_a0_db_scaling(self):
        """Test a0 specified in dB is correctly scaled to absolute magnitude."""
        a0_abs = 1e6
        opamp_a = LibraryOpAmp(a0=a0_abs)
        for a0_db in ["120 dB", "120dB", "120 db", "120db", "120 DB", "120DB", "120.0 dB"]:
            with self.subTest(a0_db):
                opamp_b = LibraryOpAmp(a0=a0_db)
                self.assertEqual(opamp_a.a0, opamp_b.a0)
