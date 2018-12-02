"""LISO output parser tests"""

import unittest
import tempfile

from zero.liso import LisoOutputParser, LisoParserError


class LisoOutputParserTestCase(unittest.TestCase):
    """Base test case class for output parser"""
    def setUp(self):
        self.reset()

    def reset(self):
        """Reset output parser"""
        self.parser = LisoOutputParser()


class ParserReuseTestCase(LisoOutputParserTestCase):
    """Test reusing output parser for the same or different circuits"""
    CIRCUIT1 = """#
                 1        9.263522296        68.32570322
                10         28.4194282        72.77463838
               100        39.53081597        19.99386514
              1000        40.08145198       0.9824328213
             10000        39.96892382       -10.87320803
            100000        34.16115885       -75.92584954
#2 capacitors:
#  0 c1 10 uF GND n1
#  1 c2 47 pF nm nout
#1 op-amp:
#  0 o1 lt1124 '+'=nin '-'=nm 'out'=nout a0=15M gbw=14.6 MHz
#       un=2.7 nV/sqrt(Hz) uc=2.3 Hz in=300 fA/sqrt(Hz) ic=100 Hz
#       umax=12 V imax=20 mA sr=4.5 V/us delay=18.9 ns
#       pole at 200 kHz (real)        pole at 200 kHz (real)        zero at 800 kHz (real)        zero at 800 kHz (real)        zero at 9.4 MHz (real)
#2 resistors:
#  0 r1 430 Ohm n1 nm
#  1 r2 43 kOhm nm nout
#4 nodes:
#  0 n1
#  1 nm
#  2 nout
#  3 nin
#Logarithmic frequency scale from 1 Hz to 100 kHz in 6 steps.
#Resistor scale factor is 4.3 kOhm
#Capacitor scale factor is 21.6795 nF
#Inductance scale factor is 400.854 mH
#Voltage scale factor is 1 V
#Current scale factor is 232.558 uA
#Frequency scale factor is 10.7271 kHz
#Voltage input at node nin, impedance 0 Ohm
#OUTPUT 1 voltage outputs:
#  0 node: nout dB Degrees
"""

    CIRCUIT2 = """#
#** Path to fil.ini is /home/sean/Workspace/LISO/c/filter/fil.ini
#** Path to opamp.lib is /home/sean/Workspace/LISO/c/filter/opamp.lib
                 1    7.186579062e-09    2.660923163e-08    1.424940889e-08                  0    1.296434102e-07
                10    6.940470296e-08    2.660967347e-08    7.893829282e-08                  0    4.278517507e-08
               100    2.496012708e-07     2.66149704e-08    2.587273733e-07                  0    1.824729154e-08
              1000    2.659386603e-07    2.661207587e-08    2.728570808e-07                  0    1.353108185e-08
             10000     2.62515383e-07    2.625171812e-08    2.690666634e-07                  0    1.279015609e-08
            100000    1.345031734e-07    1.345031827e-08    1.378563264e-07                  0     6.52389594e-09
#2 capacitors:
#  0 c1 10 uF GND n1
#  1 c2 47 pF nm nout
#1 op-amp:
#  0 o1 lt1124 '+'=nin '-'=nm 'out'=nout a0=15M gbw=14.6 MHz
#       un=2.7 nV/sqrt(Hz) uc=2.3 Hz in=300 fA/sqrt(Hz) ic=100 Hz
#       umax=12 V imax=20 mA sr=4.5 V/us delay=18.9 ns
#       pole at 200 kHz (real)        pole at 200 kHz (real)        zero at 800 kHz (real)        zero at 800 kHz (real)        zero at 9.4 MHz (real)
#2 resistors:
#  0 r1 430 Ohm n1 nm
#  1 r2 43 kOhm nm nout
#4 nodes:
#  0 n1
#  1 nm
#  2 nout
#  3 nin
#Logarithmic frequency scale from 1 Hz to 100 kHz in 6 steps.
#Resistor scale factor is 4.3 kOhm
#Capacitor scale factor is 21.6795 nF
#Inductance scale factor is 400.854 mH
#Voltage scale factor is 1 V
#Current scale factor is 232.558 uA
#Frequency scale factor is 10.7271 kHz
#Noise is computed at node nout for (nnoise=5, nnoisy=5) :
#  r1 r2 o1(U) o1(I+) o1(I-)
#Voltage input at node nin, impedance 0 Ohm
#OUTPUT 5 noise voltages caused by:
#r1 r2 o1(0) o1(1) o1(2)
"""

    def test_parser_reuse_for_different_circuit(self):
        """Test reusing output parser for different circuits"""
        # parse first circuit
        self.parser.parse(self.CIRCUIT1)
        _ = self.parser.solution()

        # parse second circuit with same parser, but with reset state
        self.parser.reset()
        self.parser.parse(self.CIRCUIT2)
        sol2a = self.parser.solution()

        # parse second circuit using a newly instantiated parser
        self.reset()
        self.parser.parse(self.CIRCUIT2)
        sol2b = self.parser.solution()

        self.assertTrue(sol2a.equivalent_to(sol2b))

    def test_parser_reuse_for_same_circuit(self):
        """Test reusing input parser for same circuits"""
        circuit = self.CIRCUIT1.splitlines()
        mid = len(circuit) // 2
        circuita = "\n".join(circuit[:mid]) + "\n" # blank line needed
        circuitb = "\n".join(circuit[mid:]) + "\n" # blank line needed

        # parse first and second parts together
        self.parser.parse(self.CIRCUIT1)
        sol1a = self.parser.solution()

        # parse first and second parts subsequently
        self.reset()
        self.parser.parse(circuita)
        self.parser.parse(circuitb)
        sol1b = self.parser.solution()

        self.assertTrue(sol1a.equivalent_to(sol1b))
