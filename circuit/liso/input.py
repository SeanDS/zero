import numpy as np
import logging

from ..format import SIFormatter
from .base import LisoParser

LOGGER = logging.getLogger("liso")

class LisoInputFormatException(Exception):
    pass


class LisoInputParser(LisoParser):
    """LISO input file parser

    This implements a lexer to identify appropriate definitions in a LISO input file,
    and a parser to build a circuit from what is found.
    """

    # top level tokens
    tokens = [
        'CHUNK',
        'NEWLINE'
    ]

    # line text
    t_CHUNK = r'[a-zA-Z0-9_=.:]+'

    # ignore spaces and tabs
    t_ignore = ' \t'

    # ignore comments
    t_ignore_COMMENT = r'\#.*'

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
    
    def p_instruction_list(self, p):
        '''instruction_list : instruction
                            | instruction_list instruction'''
        # no nothing
        pass

    # match instruction on their own lines
    def p_instruction(self, p):
        '''instruction : tokens NEWLINE
                       | NEWLINE'''
        # only when we find tokens
        if len(p) == 2:
            # skip empty line
            return
        
        p[0] = p[1]

        self.parse_instruction(p[0])

    def p_tokens(self, p):
        '''tokens : CHUNK
                  | tokens CHUNK'''
        p[0] = p[1]

        if len(p) == 3:
            p[0] += " " + p[2]

    def p_error(self, p):
        if p:
            error_msg = "LISO syntax error '%s' at line %i" % (p.value, self.lineno)
        else:
            error_msg = "LISO syntax error at end of file"
        
        raise LisoInputFormatException(error_msg)

    def parse_instruction(self, text):
        """Parses the specified text as a LISO instruction"""

        # split using spaces
        chunks = text.split()

        ident = chunks[0].lower()
        params = chunks[1:]

        if ident in ["r", "c", "l"]:
            # resistor, capacitor or inductor
            return self.parse_passive(ident, *params)
        elif ident == "op":
            # op-amp
            return self.parse_opamp(*params)
        elif ident == "freq":
            # frequency vector
            return self.parse_frequencies(*params)
        elif ident == "uinput":
            # voltage input
            return self.parse_voltage_input(*params)
        elif ident == "iinput":
            # current input
            return self.parse_current_input(*params)
        elif ident == "uoutput":
            # voltage output
            return self.parse_voltage_output(*params)
        elif ident == "ioutput":
            # current output
            return self.parse_current_output(*params)
        elif ident in ["noise"]:
            # noise input
            return self.parse_noise(*params)

        raise SyntaxError

    def parse_passive(self, passive_type, *params):
        if len(params) != 4:
            # TODO: add the error
            raise LisoInputFormatException("LISO syntax error")

        arg_names = ["name", "value", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, params)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        raise SyntaxError

    def parse_opamp(self, *params):
        arg_names = ["name", "model", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, params)}
        
        # parse extra arguments, e.g. "sr=38e6", into dict params
        if len(params) > 5:
            kwargs = {**kwargs, **self._parse_op_amp_overrides(params[5:])}

        self.circuit.add_library_opamp(**kwargs)

    @classmethod
    def _parse_op_amp_overrides(cls, overrides):
        """Parses op-amp override strings from input file
        
        In LISO, op-amp parameters can be overridden by specifying a library parameter
        after the standard op-amp definition, e.g. "op u1 ad829 gnd n1 n2 sr=38e6"
        """

        extra_args = {}

        for override in overrides:
            try:
                key, value = override.split("=")
            except ValueError:
                raise SyntaxError
            
            if key not in cls.OP_OVERRIDE_MAP.keys():
                raise SyntaxError("unknown op-amp parameter override '%s'" % key)
            
            extra_args[cls.OP_OVERRIDE_MAP[key]] = value
        
        return extra_args

    def parse_frequencies(self, *params):
        if len(params) != 4:
            # TODO: add the error
            raise LisoInputFormatException("LISO syntax error")
        
        scale = params[0].lower()
        start, _ = SIFormatter.parse(params[1])
        stop, _ = SIFormatter.parse(params[2])
        # LISO simulates specified steps + 1
        count = int(params[3]) + 1

        if scale == "lin":
            self.frequencies = np.linspace(start, stop, count)
        elif scale == "log":
            self.frequencies = np.logspace(np.log10(start), np.log10(stop),
                                           count)
        else:
            raise SyntaxError

    def parse_voltage_input(self, *params):
        if len(params) < 1:
            raise SyntaxError
        
        self.input_type = "voltage"

        # we always have at least a positive node
        self.input_node_p = params[0]

        if len(params) > 3:
            raise SyntaxError
        elif len(params) == 3:
            # floating input
            self.input_node_n = params[1]
            self.input_impedance = params[2]
        elif len(params) == 2:
            self.input_impedance = params[1]
        else:
            # default
            self.input_impedance = 50

    def parse_current_input(self, *params):
        if len(params) < 1:
            raise SyntaxError
        
        self.input_type = "current"

        # only a positive node
        self.input_node_p = params[0]
        self.input_node_n = None

        if len(params) > 2:
            raise SyntaxError
        elif len(params) == 2:
            self.input_impedance = params[1]
        else:
            # default
            self.input_impedance = 50

    def parse_voltage_output(self, *params):
        if len(params) < 1:
            raise SyntaxError

        # transfer function output
        self.output_type = "tf"

        for param in params:
            # split option by colon, ignore scaling
            output = param.split(":")[0].lower()

            # option is node
            if output == "all":
                # all outputs requested
                self.output_all_nodes = True
            elif output == "allop":
                # all op-amp outputs requested
                self.output_all_opamp_nodes = True
            elif self.output_all_nodes:
                LOGGER.warning("output node %s already requested with \"all\"" % output)
            else:
                self.output_nodes.add(output)

    def parse_current_output(self, *params):
        if len(params) < 1:
            raise SyntaxError

        # transfer function output
        self.output_type = "tf"

        for param in params:
            # split option by colon, ignore scaling
            output = param.split(":")[0].lower()
               
            # option is component
            if output == "all":
                # all outputs requested
                self.output_all_components = True
            elif output == "allop":
                # all op-amp outputs requested
                self.output_all_opamp_components = True
            elif self.output_all_components:
                LOGGER.warning("output component %s already requested with \"all\"" % output)
            else:
                self.output_components.add(output)

    def parse_noise(self, *params):
        if len(params) < 2:
            raise SyntaxError
        
        # noise output
        self.output_type = "noise"

        # noise input
        self.noise_node = params[1]

        if len(params) > 2:
            LOGGER.warning("ignoring plot options in noise command")