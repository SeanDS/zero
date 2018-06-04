import numpy as np
import logging

from ..format import SIFormatter
from .base import LisoParser, LisoOutputVoltage, LisoOutputCurrent, LisoNoiseSource

LOGGER = logging.getLogger("liso")

class LisoInputParser(LisoParser):
    """LISO input file parser

    This implements a lexer to identify appropriate definitions in a LISO input file,
    and a parser to build a circuit from what is found.
    """

    # dict mapping LISO op-amp parameter overrides to `circuit.components.OpAmp` arguments
    OP_OVERRIDE_MAP = {
        "a0": "a0",
        "gbw": "gbw",
        "delay": "delay",
        "un": "v_noise",
        "uc": "v_corner",
        "in": "i_noise",
        "ic": "i_corner",
        "umax": "v_max",
        "imax": "i_max",
        "sr": "slew_rate"
    }

    # reserved keyword tokens
    reserved = {
        "r": "R",
        "c": "C",
        "l": "L",
        "op": "OP",
        "freq": "FREQ",
        "uinput": "UINPUT",
        "iinput": "IINPUT",
        "uoutput": "UOUTPUT",
        "ioutput": "IOUTPUT",
        "noise": "NOISE"
    }

    # top level tokens
    tokens = [
        'CHUNK',
        'NEWLINE'
    ]

    # add reserved tokens
    tokens += reserved.values()

    # ignore spaces and tabs
    t_ignore = ' \t'

    # ignore comments
    t_ignore_COMMENT = r'\#.*'

    def __init__(self, *args, **kwargs):
        self._instructions = []

        super().__init__(*args, **kwargs)

    # detect new lines
    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        self._previous_newline_position = t.lexer.lexpos

        # generate newline token
        t.type = "NEWLINE"

        return t

    def t_CHUNK(self, t):
        r'[a-zA-Z0-9_=.:]+'

        # check if chunk is a keyword
        t.type = self.reserved.get(t.value.lower(), 'CHUNK')
        
        return t

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        raise SyntaxError("Illegal character '%s' on line %i at position %i" %
                          (t.value[0], self.lineno, t.lexer.lexpos - self._previous_newline_position))

    def p_instruction_list(self, p):
        '''instruction_list : instruction
                            | instruction_list instruction'''
        pass

    # match instruction on their own lines
    def p_instruction(self, p):
        '''instruction : resistor NEWLINE
                       | capacitor NEWLINE
                       | inductor NEWLINE
                       | opamp NEWLINE
                       | freq NEWLINE
                       | uinput NEWLINE
                       | iinput NEWLINE
                       | uoutput NEWLINE
                       | ioutput NEWLINE
                       | noise NEWLINE
                       | NEWLINE'''
        pass

    def p_resistor(self, p):
        '''resistor : R CHUNK CHUNK CHUNK CHUNK'''
        
        # parse as resistor
        self.parse_passive("r", *p[2:])

    def p_capacitor(self, p):
        '''capacitor : C CHUNK CHUNK CHUNK CHUNK'''
        
        # parse as capacitor
        self.parse_passive("c", *p[2:])

    def p_inductor(self, p):
        '''inductor : L CHUNK CHUNK CHUNK CHUNK'''
        
        # parse as inductor
        self.parse_passive("l", *p[2:])

    def p_opamp(self, p):
        '''opamp : OP CHUNK CHUNK CHUNK CHUNK CHUNK
                 | OP CHUNK CHUNK CHUNK CHUNK CHUNK chunks'''
        
        # parse as op-amp
        self.parse_library_opamp(*p[2:])

    def p_freq(self, p):
        '''freq : FREQ CHUNK CHUNK CHUNK CHUNK'''

        # parse frequencies
        self.parse_frequencies(*p[2:])

    def p_uinput(self, p):
        '''uinput : UINPUT CHUNK
                  | UINPUT CHUNK CHUNK
                  | UINPUT CHUNK CHUNK CHUNK'''

        # parse voltage input
        self.parse_voltage_input(*p[2:])

    def p_iinput(self, p):
        '''iinput : IINPUT CHUNK CHUNK'''

        # parse current input
        self.parse_current_input(*p[2:])

    def p_uoutput(self, p):
        '''uoutput : UOUTPUT chunks'''

        # parse voltage outputs
        self.parse_voltage_output(*p[2:])

    def p_ioutput(self, p):
        '''ioutput : IOUTPUT chunks'''

        # parse current outputs
        self.parse_current_output(*p[2:])

    def p_noise(self, p):
        '''noise : NOISE chunks'''

        # parse noise node
        self.parse_noise_output(*p[2:])

    def p_chunks(self, p):
        '''chunks : CHUNK
                  | chunks CHUNK'''
        p[0] = p[1]

        if len(p) == 3:
            p[0] += " " + p[2]

    def p_error(self, p):
        if p:
            message = "LISO syntax error '%s' at line %i" % (p.value, self.lineno)
        else:
            message = "LISO syntax error at end of file"
        
        raise SyntaxError(message)

    def parse_passive(self, passive_type, *params):
        if len(params) != 4:
            # TODO: add the error
            raise SyntaxError("LISO syntax error")

        arg_names = ["name", "value", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, params)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        raise SyntaxError

    def parse_library_opamp(self, *params):
        if len(params) < 5 or len(params) > 6:
            raise SyntaxError
        
        arg_names = ["name", "model", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, params)}
        
        if len(params) == 6:
            # parse extra arguments, e.g. "sr=38e6", into dict params
            kwargs = {**kwargs, **self._parse_op_amp_overrides(params[5])}

        self.circuit.add_library_opamp(**kwargs)

    @classmethod
    def _parse_op_amp_overrides(cls, override_list):
        """Parses op-amp override strings from input file
        
        In LISO, op-amp parameters can be overridden by specifying a library parameter
        after the standard op-amp definition, e.g. "op u1 ad829 gnd n1 n2 sr=38e6"
        """

        extra_args = {}

        for override in override_list.split():
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
            raise SyntaxError("invalid frequency definition")
        
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

    def parse_voltage_output(self, params):
        if len(params) < 1:
            raise SyntaxError
        
        # transfer function output
        self.output_type = "tf"

        for param in params.split():
            # split option by colon
            self.add_voltage_output(param)

    def add_voltage_output(self, param_str):
        params = param_str.split(":")
        node_name = params[0]
        scales = params[1:]

        if node_name.lower() == "all":
            output = LisoOutputVoltage(scales=scales, all_nodes=True)
        elif node_name.lower() == "allop":
            output = LisoOutputVoltage(scales=scales, all_opamps=True)
        else:
            output = LisoOutputVoltage(node=node_name, scales=scales)

        self.tf_outputs.append(output)

    def parse_current_output(self, params):
        if len(params) < 1:
            raise SyntaxError

        # transfer function output
        self.output_type = "tf"

        for param in params.split():
            # split option by colon
            self.add_current_output(param)

    def add_current_output(self, param_str):
        params = param_str(":")
        component_name = params[0]
        scales = params[1:]

        if component_name.lower() == "all":
            output = LisoOutputCurrent(scales=scales, all_components=True)
        elif component_name.lower() == "allop":
            output = LisoOutputCurrent(scales=scales, all_opamps=True)
        else:
            output = LisoOutputCurrent(component=component_name, scales=scales)

        self.tf_outputs.append(output)

    def parse_noise_output(self, *params):
        if len(params) < 2:
            raise SyntaxError
        
        if len(params) > 2:
            LOGGER.warning("ignoring noise source options in noise command")

        nodes = []

        # noise output
        self.output_type = "noise"

        noise_node = params[0].lower()
        
        # TODO: loop over rest of params
        noise_input = params[1].lower()

        if noise_node == "all":
            pass
        elif noise_node == "allr":
            pass
        elif noise_node == "allop":
            pass
        elif noise_node == "sum":
            raise NotImplementedError
        else:
            nodes.append(LisoNoiseSource(self.circuit.get_node(noise_node)))

        # noise input
        #self.noise_sources.extend(nodes)
    
        # noise output
        self.noise_output_node = self.circuit.get_node(noise_node)