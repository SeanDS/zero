"""LISO input parser tests"""

import unittest
import tempfile

from zero.liso import LisoInputParser, LisoParserError


class LisoInputParserTestCase(unittest.TestCase):
    """Base test case class for input parser"""
    def setUp(self):
        self.reset()

    def reset(self):
        """Reset input parser"""
        self.parser = LisoInputParser()

class InvalidFileTestCase(LisoInputParserTestCase):
    """Voltage output command tests"""
    def test_empty_string(self):
        """Test empty file"""
        self.parser.parse("")
        self.assertRaisesRegex(LisoParserError, "no circuit defined", self.parser.solution)

    def test_empty_file(self):
        """Test empty file"""
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            # write empty line
            fp.write("")
            # parse
            self.parser.parse(path=fp.name)

        self.assertRaisesRegex(LisoParserError, "no circuit defined", self.parser.solution)

    def test_file_with_blank_line(self):
        """Test file with only a blank line"""
        with tempfile.NamedTemporaryFile(mode="w") as fp:
            # write empty line
            fp.write("\n")
            # parse
            self.parser.parse(path=fp.name)

        self.assertRaisesRegex(LisoParserError, "no circuit defined", self.parser.solution)


class ParserReuseTestCase(LisoInputParserTestCase):
    """Test reusing input parser for the same or different circuits"""
    def test_parser_reuse_for_different_circuit(self):
        """Test reusing input parser for different circuits"""
        circuit1 = """
r r1 1k n1 n2
op op1 OP00 gnd n2 n3
r r2 2k n2 n3
freq log 1 1k 100
uinput n1
uoutput n3
"""

        circuit2 = """
r r1 2k n1 n2
op op1 OP00 gnd n2 n3
r r2 4k n2 n3
freq log 1 2k 100
uinput n1
uoutput n3
"""

        # parse first circuit
        self.parser.parse(circuit1)
        _ = self.parser.solution()

        # parse second circuit with same parser, but with reset state
        self.parser.reset()
        self.parser.parse(circuit2)
        sol2a = self.parser.solution()

        # parse second circuit using a newly instantiated parser
        self.reset()
        self.parser.parse(circuit2)
        sol2b = self.parser.solution()

        self.assertTrue(sol2a.equivalent_to(sol2b))

    def test_parser_reuse_for_same_circuit(self):
        """Test reusing input parser for same circuits"""
        circuit1a = """
r r1 1k n1 n2
op op1 OP00 gnd n2 n3
r r2 2k n2 n3
"""

        circuit1b = """
freq log 1 1k 100
uinput n1
uoutput n3
"""

        # parse first and second parts together
        self.parser.parse(circuit1a + circuit1b)
        sol1a = self.parser.solution()

        # parse first and second parts subsequently
        self.reset()
        self.parser.parse(circuit1a)
        self.parser.parse(circuit1b)
        sol1b = self.parser.solution()

        self.assertTrue(sol1a.equivalent_to(sol1b))


class VoltageOutputTestCase(LisoInputParserTestCase):
    """Voltage output command tests"""
    def test_invalid_output_node(self):
        """Test nonexistent output node"""
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# n3 doesn't exist
uoutput n3
""")
        self.assertRaisesRegex(LisoParserError,
                               r"output element 'n3' is not present in the circuit",
                               self.parser.solution)

    def test_output_all(self):
        """Test output all node and component voltages"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 0
uoutput no all
""")
        self.parser.build()
        # there should be 8 outputs
        self.assertEqual(8, self.parser.n_response_outputs)

    def test_output_allop(self):
        """Test output all op-amp node and component voltages"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 0
uoutput no ni allop
""")
        self.parser.build()
        # 3 op-amp outputs, one of which is no, plus ni
        self.assertEqual(4, self.parser.n_response_outputs)


class CurrentOutputTestCase(LisoInputParserTestCase):
    """Current output command tests"""
    def test_invalid_output_node(self):
        """Test nonexistent output component"""
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# r2 doesn't exist
ioutput r2
""")
        self.assertRaisesRegex(LisoParserError,
                               r"output element 'r2' is not present in the circuit",
                               self.parser.solution)

    def test_output_all(self):
        """Test output all component currents"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 0
ioutput load all
""")
        self.parser.build()
        # There should be 15 outputs corresponding to the 15 components in the circuit.
        self.assertEqual(15, self.parser.n_response_outputs)

    def test_output_allop(self):
        """Test output all op-amp currents"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 0
ioutput load ri1 allop
""")
        self.parser.build()
        # 3 op-amp outputs, plus two independent
        self.assertEqual(5, self.parser.n_response_outputs)


class NoiseOutputTestCase(LisoInputParserTestCase):
    """Noise output command tests"""
    def test_invalid_noise_output_element(self):
        """Test nonexistent noise output node"""
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# n3 doesn't exist
noise n3 all
""")
        self.assertRaisesRegex(LisoParserError,
                               r"noise output element 'n3' is not present in the circuit",
                               self.parser.solution)

    def test_invalid_noisy_node(self):
        """Test nonexistent noisy node"""
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# r2 doesn't exist
noise n2 r2
""")
        self.assertRaisesRegex(LisoParserError,
                               r"noise source 'r2' is not present in the circuit",
                               self.parser.solution)

        self.reset()

        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
noise n2 all
# r2 doesn't exist
noisy r2
""")
        self.assertRaisesRegex(LisoParserError,
                               r"noise source 'r2' is not present in the circuit",
                               self.parser.solution)

    def test_output_all(self):
        """Test output all node and component noise"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 50
noise no r1 r2 all
""")
        self.parser.build()
        # there are 15 components that produce noise; two are duplicated
        self.assertEqual(15, self.parser.n_displayed_noise)


    def test_output_allop(self):
        """Test output all op-amp node and component noise"""
        self.parser.parse("""
c c1 270.88486n nm2 no
c c2 7.516u no3 np1
c ca 1.484u nm2 ni
op n1a OP27 np1 gnd no1
op n2a OP27 gnd nm2 no
op n3a OP27 gnd nm3 no3
r ri2 2.6924361k nm3 no1
r ri1 1k no3 nm3
r r1 40.246121k no1 nm2
r r2 2.0651912k np1 no
r ra 115.16129k nm2 ni
r rb 2.0154407k np1 ni
r rq 22.9111k nm2 no
r load 1k no gnd
r rin 50 nii ni
freq log 1 100 10
uinput nii 50
noise no rin n2a allop
""")
        self.parser.build()
        # there should be 2 noise outputs per op-amp, so 6 total, plus one extra; one is duplicate
        self.assertEqual(7, self.parser.n_displayed_noise)

    def test_cannot_compute_noise_sum_without_noisy_command(self):
        self.parser.parse("""
r r1 1k n1 n2
r r2 10k n2 n3
op op1 op00 gnd n2 n3
freq log 1 100 10
uinput n1
noise op1 sum
""")

        # noise sum requires noisy command
        self.assertRaisesRegex(LisoParserError,
                               r"noise sum requires noisy components to be defined",
                               self.parser.solution)
