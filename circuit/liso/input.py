import numpy as np
import logging

from ..format import SIFormatter
from .base import LisoParser, LisoOutputElement, LisoNoiseSource

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
        "un": "vn",
        "uc": "vc",
        "in": "in",
        "ic": "ic",
        "umax": "vmax",
        "imax": "imax",
        "sr": "slew_rate"
    }

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

    def __init__(self, *args, **kwargs):
        self._instructions = []

        super(LisoInputParser, self).__init__(*args, **kwargs)

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
        raise SyntaxError("Illegal character '%s' on line %i at position %i" %
                          (t.value[0], self.lineno, t.lexer.lexpos - self._previous_newline_position))

    def p_instruction_list(self, p):
        '''instruction_list : instruction
                            | instruction_list instruction'''
        pass

    # match instruction on their own lines
    def p_instruction(self, p):
        '''instruction : tokens NEWLINE
                       | NEWLINE'''
        # only when we find tokens
        if len(p) == 2:
            # skip empty line
            return
        
        instruction = p[1]
        p[0] = instruction

        self.parse_instruction(instruction)

    def p_tokens(self, p):
        '''tokens : CHUNK
                  | tokens CHUNK'''
        p[0] = p[1]

        if len(p) == 3:
            p[0] += " " + p[2]

    def p_error(self, p):
        if p:
            message = "LISO syntax error '%s' at line %i" % (p.value, self.lineno)
        else:
            message = "LISO syntax error at end of file"
        
        raise SyntaxError(message)

    def build(self):
        # add input component, if not yet present
        self._set_circuit_input()

    def parse_instruction(self, instruction):
        """Parses the specified text as a LISO input file instruction"""

        # split using spaces
        chunks = instruction.split()

        ident = chunks[0].lower()
        params = chunks[1:]

        if ident == "r":
            # resistor
            return self.parse_passive("r", *params)
        elif ident == "c":
            # capacitor
            return self.parse_passive("c", *params)
        elif ident == "l":
            # inductor
            return self.parse_passive("l", *params)
        elif ident == "op":
            # op-amp
            return self.parse_library_opamp(*params)
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
            return self.parse_noise_output(*params)

        raise SyntaxError("LISO syntax error")

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

    def parse_voltage_output(self, *params):
        if len(params) < 1:
            raise SyntaxError

        # transfer function output
        self.output_type = "tf"

        for param in params:
            # split option by colon
            self.add_voltage_output(param.split(":"))

    def add_voltage_output(self, params):
        node_name = params[0].lower()
        scales = params[1:]

        if node_name == "all":
            outputs = [LisoOutputElement(node, scales) for node in self.circuit.non_gnd_nodes]
        elif node_name == "allop":
            outputs = [LisoOutputElement(node, scales) for node in self.circuit.opamp_output_nodes]
        else:
            outputs = [LisoOutputElement(self.circuit.get_node(node_name), scales)]

        self.tf_outputs.extend(outputs)

    def parse_current_output(self, *params):
        if len(params) < 1:
            raise SyntaxError

        # transfer function output
        self.output_type = "tf"

        for param in params:
            # split option by colon
            self.add_current_output(param.split(":"))

    def add_current_output(self, params):
        component_name = params[0].lower()
        scales = params[1:]

        if component_name == "all":
            outputs = [LisoOutputElement(component, scales) for component in self.circuit.components]
        elif component_name == "allop":
            outputs = [LisoOutputElement(component, scales) for component in self.circuit.opamps]
        else:
            outputs = [LisoOutputElement(self.circuit.get_component(component_name), scales)]

        self.tf_outputs.extend(outputs)

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