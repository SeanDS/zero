import numpy as np
import logging
import re

from ..format import SIFormatter
from .base import LisoParser

LOGGER = logging.getLogger("liso")

class LisoOutputFormatException(Exception):
    pass


class LisoOutputParser(LisoParser):
    """LISO output file parser

    This implements a lexer to identify appropriate definitions in a LISO output file,
    and a parser to build a solution and circuit from what is found.
    """

    # additional states
    states = (
        ('resistors', 'inclusive'),
        ('capacitors', 'inclusive'),
        ('inductors', 'inclusive'),
        ('opamps', 'inclusive'),
        ('nodes', 'inclusive'),
    )

    # data lexer tokens
    tokens = [
        'DATUM',
        'NEWLINE',
        'RESISTOR',
        'CAPACITOR',
        'INDUCTOR',
        'OPAMP_CHUNK_1',
        'OPAMP_CHUNK_2',
        'OPAMP_CHUNK_3',
        'OPAMP_CHUNK_4',
        'NODE'
    ]

    # data point
    t_DATUM = r'-?(\d+\.\d*|\d*\.\d+|\d+)([eE][-]?\d*\.?\d*)?'

    # comment (including circuit definitions)
    t_ignore_COMMENT = r'\#.*'

    # ignore spaces and tabs
    t_ignore = ' \t'

    def __init__(self, *args, **kwargs):
        self._data = None

        # number of each element as reported by the output file
        self.nresistors = None
        self.ncapacitors = None
        self.ninductors = None
        self.nopamps = None
        self.nnodes = None

        super(LisoOutputParser, self).__init__(*args, **kwargs)

    def parse(self, *args, **kwargs):
        # create empty data set
        self._data = []

        super(LisoOutputParser, self).parse(*args, **kwargs)

        # TODO: create solution from parsed data
        
        # clear data
        self._data = None

    def t_ANY_resistors(self, t):
        r'\#(?P<n>\d+)\sresistors?:'
        self.nresistors = t.lexer.lexmatch.group('n')
        t.lexer.begin('resistors')

    def t_ANY_capacitors(self, t):
        r'\#(?P<n>\d+)\scapacitors?:'
        self.ncapacitors = t.lexer.lexmatch.group('n')
        t.lexer.begin('capacitors')

    def t_ANY_inductors(self, t):
        r'\#(?P<n>\d+)\sinductors?:'
        self.ninductors = t.lexer.lexmatch.group('n')
        t.lexer.begin('inductors')

    def t_ANY_opamps(self, t):
        r'\#(?P<n>\d+)\sop-amps?:'
        self.nopamps = t.lexer.lexmatch.group('n')
        t.lexer.begin('opamps')

    def t_ANY_nodes(self, t):
        r'\#(?P<n>\d+)\snodes?:'
        self.nnodes = t.lexer.lexmatch.group('n')
        t.lexer.begin('nodes')

    def t_resistors_RESISTOR(self, t):
        r'\#\s+\d+\s+(?P<resistor>.*)'
        t.type = "RESISTOR"
        t.value = t.lexer.lexmatch.group('resistor')
        return t

    def t_capacitors_CAPACITOR(self, t):
        r'\#\s+\d+\s+(?P<capacitor>.*)'
        t.type = "CAPACITOR"
        t.value = t.lexer.lexmatch.group('capacitor')
        return t

    def t_inductors_INDUCTOR(self, t):
        r'\#\s+\d+\s+(?P<inductor>.*)'
        t.type = "INDUCTOR"
        t.value = t.lexer.lexmatch.group('inductor')
        return t

    def t_opamps_OPAMP_CHUNK_1(self, t):
        r'\#\s+\d+\s+(?P<opamp1>.*)'
        t.type = "OPAMP_CHUNK_1"
        t.value = t.lexer.lexmatch.group('opamp1')
        return t

    def t_opamps_OPAMP_CHUNK_2(self, t):
        r'\#\s+(?P<opamp2>un=.*)'
        t.type = "OPAMP_CHUNK_2"
        t.value = t.lexer.lexmatch.group('opamp2')
        return t

    def t_opamps_OPAMP_CHUNK_3(self, t):
        r'\#\s+(?P<opamp3>umax=.*)'
        t.type = "OPAMP_CHUNK_3"
        t.value = t.lexer.lexmatch.group('opamp3')
        return t

    def t_opamps_OPAMP_CHUNK_4(self, t):
        r'\#\s+(?P<opamp4>pole.*)'
        t.type = "OPAMP_CHUNK_4"
        t.value = t.lexer.lexmatch.group('opamp4')
        return t

    def t_nodes_NODE(self, t):
        r'\#\s+\d+\s+(?P<node>.*)'
        t.type = "NODE"
        t.value = t.lexer.lexmatch.group('node')
        return t

    # detect new lines
    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        self._previous_newline_position = t.lexer.lexpos

        # generate newline token
        t.type = "NEWLINE"

        return t

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        print("Illegal character '%s' on line %i at position %i" %
              (t.value[0], self.lineno, t.lexer.lexpos - self._previous_newline_position))

        # skip forward a character
        t.lexer.skip(1)
    
    def p_file_contents(self, p):
        '''file_contents : file_line
                         | file_contents file_line'''
        # do nothing
        pass

    def p_file_line(self, p):
        # a line of data or a comment line
        '''file_line : data_line
                     | metadata_line'''

    def p_data_line(self, p):
        # list of measurements on a line of its own
        '''data_line : data NEWLINE
                     | NEWLINE'''
        # only when we find tokens
        if len(p) == 2:
            # skip empty line
            return
        
        # add new row to data
        self._data.append(p[1])
        print("saved", p[1])

    def p_data(self, p):
        # list of measurements
        '''data : data DATUM
                | DATUM'''
                
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_metadata_line(self, p):
        # metadata on its own line, e.g. comment or resistor definition
        '''metadata_line : RESISTOR NEWLINE
                         | CAPACITOR NEWLINE
                         | INDUCTOR NEWLINE
                         | opamp
                         | NODE NEWLINE'''
        
        p[0] = p[1]
        print("BRRP:", p[1])

    def p_opamp(self, p):
        # join chunks of op-amp definition together
        '''opamp : OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE OPAMP_CHUNK_4 NEWLINE
                 | OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE'''
        # join without newlines
        p[0] = " ".join(p[1::2])

    def p_error(self, p):
        if p:
            error_msg = "LISO syntax error '%s' at line %i" % (p.value, self.lineno)
        else:
            error_msg = "LISO syntax error at end of file"
        
        print(error_msg)
        #raise LisoOutputFormatException(error_msg)
