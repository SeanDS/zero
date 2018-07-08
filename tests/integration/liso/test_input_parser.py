"""LISO input parser tests"""

import unittest

from circuit.liso import LisoInputParser, LisoParserError

class LisoInputParserTestCase(unittest.TestCase):
    """Base test case class for input parser"""
    def setUp(self):
        self.reset()

    def reset(self):
        """Reset input parser"""
        self.parser = LisoInputParser()

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
        self.assertEqual(8, self.parser.n_tf_outputs)

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
        self.assertEqual(4, self.parser.n_tf_outputs)

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
        # there should be 16 outputs (15 components above plus input component)
        self.assertEqual(16, self.parser.n_tf_outputs)

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
        self.assertEqual(5, self.parser.n_tf_outputs)

class NoiseOutputTestCase(LisoInputParserTestCase):
    """Noise output command tests"""
    def test_invalid_noise_output_node(self):
        """Test nonexistent noise output node"""
        self.parser.parse("""
r r1 1k n1 n2
uinput n1
freq log 1 1k 100
# n3 doesn't exist
noise n3 all
""")
        self.assertRaisesRegex(LisoParserError,
                               r"noise output node 'n3' is not present in the circuit",
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
