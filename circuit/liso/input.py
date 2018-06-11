import numpy as np
import logging

from ..components import CurrentNoise, VoltageNoise, JohnsonNoise
from ..format import SIFormatter
from .base import LisoParser, LisoOutputVoltage, LisoOutputCurrent, LisoNoiseSource, LisoParserError

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

        # noise source definitions, later turned into LisoNoiseSource objects
        self._noise_source_defs = []

        super().__init__(*args, **kwargs)

    def _do_build(self, *args, **kwargs):
        super()._do_build(*args, **kwargs)

        sources = []

        # now that we have all the circuit components, create noise source objects
        for index, definition in enumerate(self._noise_source_defs):
            component = self.circuit.get_component(definition[0])

            if len(definition) > 1:
                # op-amp noise type specified
                type_str = definition[1].lower()

                if type_str == "u":
                    noise = component.voltage_noise
                elif type_str == "+":
                    # non-inverting input current noise
                    noise = component.non_inv_current_noise
                elif type_str == "-":
                    # inverting input current noise
                    noise = component.inv_current_noise
                else:
                    raise LisoParserError("unrecognised op-amp noise source '%s'" % definition[1], self.lineno)
            else:
                # must be a resistor
                noise = component.johnson_noise
        
            # add noise source
            sources.append(LisoNoiseSource(noise=noise, index=index))

        self.noise_sources = sources

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
        raise LisoParserError(f"illegal character '{t.value[0]}'", self.lineno, t.lexer.lexpos - self._previous_newline_position)

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
        self.parse_voltage_output(p[2])

    def p_ioutput(self, p):
        '''ioutput : IOUTPUT chunks'''

        # parse current outputs
        self.parse_current_output(p[2])

    def p_noise(self, p):
        '''noise : NOISE chunks'''

        # parse noise node
        self.parse_noise_output(p[2])

    def p_chunks(self, p):
        '''chunks : CHUNK
                  | chunks CHUNK'''
        p[0] = p[1]

        if len(p) == 3:
            p[0] += " " + p[2]

    def p_error(self, p):
        if p:
            # check for unexpected new line
            if p.value == "\n":
                message = "unexpected end of line"
            else:
                message = "'%s'" % p.value
        else:
            message = "unexpected end of file"
        
        raise LisoParserError(message, self.lineno)

    def parse_passive(self, passive_type, *params):
        if len(params) != 4:
            raise LisoParserError("unexpected parameter count (%d)" % len(params), self.lineno)

        arg_names = ["name", "value", "node1", "node2"]
        kwargs = {name: value for name, value in zip(arg_names, params)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        raise LisoParserError(f"unrecognised passive component '{passive_type}'", self.lineno)

    def parse_library_opamp(self, *params):
        if len(params) < 5 or len(params) > 6:
            raise LisoParserError("unexpected parameter count (%d)" % len(params), self.lineno)
        
        arg_names = ["name", "model", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, params)}
        
        if len(params) == 6:
            # parse extra arguments, e.g. "sr=38e6", into dict params
            kwargs = {**kwargs, **self._parse_op_amp_overrides(params[5])}

        self.circuit.add_library_opamp(**kwargs)

    def _parse_op_amp_overrides(self, override_list):
        """Parses op-amp override strings from input file
        
        In LISO, op-amp parameters can be overridden by specifying a library parameter
        after the standard op-amp definition, e.g. "op u1 ad829 gnd n1 n2 sr=38e6"
        """

        extra_args = {}

        for override in override_list.split():
            try:
                key, value = override.split("=")
            except ValueError:
                raise LisoParserError("op-amp parameter override must be in the form 'param=value'", self.lineno)
            
            if key not in self.OP_OVERRIDE_MAP.keys():
                raise LisoParserError(f"unknown op-amp override parameter '{key}'", self.lineno)
            
            extra_args[self.OP_OVERRIDE_MAP[key]] = value
        
        return extra_args

    def parse_frequencies(self, *params):
        if len(params) != 4:
            raise LisoParserError("unexpected parameter count (%d)" % len(params), self.lineno)
        
        scale = params[0]
        start, _ = SIFormatter.parse(params[1])
        stop, _ = SIFormatter.parse(params[2])
        # LISO simulates specified steps + 1
        count = int(params[3]) + 1

        if scale.lower() == "lin":
            self.frequencies = np.linspace(start, stop, count)
        elif scale.lower() == "log":
            self.frequencies = np.logspace(np.log10(start), np.log10(stop),
                                           count)
        else:
            raise LisoParserError(f"invalid frequency scale '{scale}'", self.lineno)

    def parse_voltage_input(self, *params):
        if len(params) < 1 or len(params) > 3:
            raise LisoParserError("unexpected parameter count (%d)" % len(params), self.lineno)
        
        self.input_type = "voltage"

        # we always have at least a positive node
        self.input_node_p = params[0]

        if len(params) == 3:
            # floating input
            self.input_node_n = params[1]
            self.input_impedance = params[2]
        elif len(params) == 2:
            self.input_impedance = params[1]
        else:
            # default
            self.input_impedance = 50

    def parse_current_input(self, *params):
        if len(params) < 1 or len(params) > 2:
            raise LisoParserError("unexpected parameter count (%d)" % len(params), self.lineno)
        
        self.input_type = "current"

        # only a positive node
        self.input_node_p = params[0]
        self.input_node_n = None

        if len(params) == 2:
            self.input_impedance = params[1]
        else:
            # default
            self.input_impedance = 50

    def parse_voltage_output(self, output_str):        
        # transfer function output
        self.output_type = "tf"

        for param in output_str.split():
            # split option by colon
            self.add_voltage_output(param)

    def add_voltage_output(self, param_str):
        params = param_str.split(":")
        node_name = params[0]
        scales = params[1:]

        if node_name.lower() == "all":
            # all node voltages
            self._output_all_nodes = True
        elif node_name.lower() == "allop":
            # op-amp outputs voltages
            self._output_all_opamp_nodes = True
        else:
            # add output
            self.tf_outputs.append(LisoOutputVoltage(node=node_name, scales=scales))

    def parse_current_output(self, output_str):
        # transfer function output
        self.output_type = "tf"

        for param in output_str.split():
            # split option by colon
            self.add_current_output(param)

    def add_current_output(self, param_str):
        params = param_str.split(":")
        component_name = params[0]
        scales = params[1:]

        if component_name.lower() == "all":
            # all component currents
            self._output_all_components = True
        elif component_name.lower() == "allop":
            # op-amp output currents
            self._output_all_opamps = True
        else:
            # add output
            self.tf_outputs.append(LisoOutputCurrent(component=component_name, scales=scales))

    def parse_noise_output(self, noise_str):
        # split by whitespace
        params = noise_str.split()

        # noise output
        self.output_type = "noise"

        # noise output
        self.noise_output_node = params[0]

        if len(params) > 1:
            # parse noise sources
            for source_str in params[1:]:
                # strip any remaining whitespace
                source_str = source_str.strip()

                # split off op-amp port settings
                source_pieces = source_str.split(":")

                component_name = source_pieces[0]

                if component_name.lower() == "all":
                    # all component noises
                    self._source_all_components = True
                elif component_name.lower() == "allop":
                    self._source_all_opamps = True
                elif component_name.lower() == "allr":
                    # all resistor noises
                    self._source_all_resistors = True
                elif component_name.lower() == "sum":
                    raise NotImplementedError
                else:
                    # individual component
                    definition = [component_name]
                    
                    if len(source_pieces) > 1:
                        # add op-amp noise type
                        definition.append(source_pieces[1])
                    
                    self._noise_source_defs.append(definition)