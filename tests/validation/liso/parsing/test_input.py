import unittest

from circuit.liso.input import LisoInputParser

class LisoInputParserTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = LisoInputParser()

    def test_component_invalid_type(self):
        kwargs = {"text": """
# component type "a" doesn't exist
a c1 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)

    def test_component_missing_name(self):
        kwargs = {"text": """
# no component name given
c 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)

    def test_component_invalid_value(self):
        kwargs = {"text": """
# invalid component value
c c1 -10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout
freq log 1 100k 100
uinput nin 0
uoutput nout:db:deg
        """}

        self.assertRaises(SyntaxError, self.parser.parse, **kwargs)