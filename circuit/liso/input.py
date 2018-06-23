import numpy as np
import logging

from ..components import CurrentNoise, VoltageNoise, JohnsonNoise
from ..format import Quantity
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
        "noise": "NOISE",
        "noisy": "NOISY"
    }

    # top level tokens
    tokens = [
        'CHUNK',
        'NEWLINE',
        'EOF'
    ]

    # add reserved tokens
    tokens += reserved.values()

    # ignore spaces and tabs
    t_ignore = ' \t'

    # ignore comments
    t_ignore_COMMENT = r'\#.*'

    def __init__(self, *args, **kwargs):
        self._instructions = []

        # output flags
        self._output_all_nodes = False
        self._output_all_opamp_nodes = False
        self._output_all_components = False
        self._output_all_opamps = False

        # noise source flags
        self._source_all_components = False
        self._source_all_opamps = False
        self._source_all_resistors = False

        # noisy source flags
        self._noisy_all_components = False
        self._noisy_all_opamps = False
        self._noisy_all_resistors = False

        # noise source definitions, later turned into LisoNoiseSource objects
        self._noise_source_defs = []

        super().__init__(*args, **kwargs)

    def _do_build(self, *args, **kwargs):
        super()._do_build(*args, **kwargs)

        sources = []

        # now that we have all the circuit components, create noise source objects
        for definition in self._noise_source_defs:
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
                    self.p_error("unrecognised op-amp noise source '%s'" % definition[1])
                
                # add noise source
                sources.append(noise)
            else:
                # get all of the component's noise sources
                sources.extend(component.noise)

        self.noise_sources = sources

    @LisoParser.output_nodes.getter
    def output_nodes(self):
        nodes = set(super().output_nodes)

        if self._output_all_nodes:
            # show all nodes
            nodes.update(self.circuit.non_gnd_nodes)
        elif self._output_all_opamp_nodes:
            # show all op-amp nodes
            nodes.update(self.circuit.opamp_output_nodes)

        return nodes

    @LisoParser.output_components.getter
    def output_components(self):
        components = set(super().output_components)

        if self._output_all_components:
            # show all components
            components.update(self.circuit.components)
        elif self._output_all_opamps:
            # show all op-amps
            components.update(self.circuit.opamps)
        
        return components

    @LisoParser.noise_sources.getter
    def noise_sources(self):
        sources = set(super().noise_sources)

        if self._source_all_components:
            # show all noise sources
            sources.update(*[component.noise for component in self.circuit.components])
        elif self._source_all_opamps:
            # show all op-amp noise sources
            sources.update(*[component.noise for component in self.circuit.opamps])
        elif self._source_all_resistors:
            # show all resistor noise sources
            sources.update(*[component.noise for component in self.circuit.resistors])

        return sources

    @LisoParser.noise_sum_sources.getter
    def noise_sum_sources(self):
        sum_sources = set(super().noise_sum_sources)

        if self._noisy_all_components:
            # show all noise sources
            sum_sources.update(*[component.noise for component in self.circuit.components])
        elif self._noisy_all_opamps:
            # show all op-amp noise sources
            sum_sources.update(*[component.noise for component in self.circuit.opamps])
        elif self._noisy_all_resistors:
            # show all resistor noise sources
            sum_sources.update(*[component.noise for component in self.circuit.resistors])

        return sum_sources

    # detect new lines
    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        self._previous_newline_position = t.lexer.lexpos

        # generate newline token
        t.type = "NEWLINE"

        return t

    # detect end of file
    def t_eof(self, t):
        if self._eof:
            # EOF token already thrown
            # finish
            return None
        
        self._eof = True

        # EOF acts like a newline
        self.lineno += 1

        # throw one more EOF token
        t.type = "EOF"
        return t

    def t_CHUNK(self, t):
        r'[a-zA-Z0-9_=.:]+'

        # check if chunk is a keyword
        t.type = self.reserved.get(t.value.lower(), 'CHUNK')
        
        return t

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        raise LisoParserError("illegal character '{char}'".format(char=t.value[0]), self.lineno,
                              t.lexer.lexpos - self._previous_newline_position)

    def p_instruction_list(self, p):
        '''instruction_list : instruction
                            | instruction_list instruction'''
        pass

    def p_empty_instruction(self, p):
        '''instruction : end'''
        pass
    
    def p_end(self, p):
        '''end : NEWLINE
               | EOF'''
        pass

    def p_resistor(self, p):
        '''instruction : R CHUNK CHUNK CHUNK CHUNK end'''
        
        # parse as resistor
        self.parse_passive("r", *p[2:-1])

    def p_capacitor(self, p):
        '''instruction : C CHUNK CHUNK CHUNK CHUNK end'''
        
        # parse as capacitor
        self.parse_passive("c", *p[2:-1])

    def p_inductor(self, p):
        '''instruction : L CHUNK CHUNK CHUNK CHUNK end'''
        
        # parse as inductor
        self.parse_passive("l", *p[2:-1])

    def p_opamp(self, p):
        '''instruction : OP CHUNK CHUNK CHUNK CHUNK CHUNK end
                       | OP CHUNK CHUNK CHUNK CHUNK CHUNK chunks end'''
        
        # parse as op-amp
        self.parse_library_opamp(*p[2:-1])

    def p_freq(self, p):
        '''instruction : FREQ CHUNK CHUNK CHUNK CHUNK end'''

        # parse frequencies
        self.parse_frequencies(*p[2:-1])

    def p_uinput(self, p):
        '''instruction : UINPUT CHUNK end
                       | UINPUT CHUNK CHUNK end
                       | UINPUT CHUNK CHUNK CHUNK end'''

        # parse voltage input
        self.parse_voltage_input(*p[2:-1])

    def p_iinput(self, p):
        '''instruction : IINPUT CHUNK end
                       | IINPUT CHUNK CHUNK end'''

        # parse current input
        self.parse_current_input(*p[2:-1])

    def p_uoutput(self, p):
        '''instruction : UOUTPUT chunks end'''

        # parse voltage outputs
        self.parse_voltage_output(p[2])

    def p_ioutput(self, p):
        '''instruction : IOUTPUT chunks end'''

        # parse current outputs
        self.parse_current_output(p[2])

    def p_noise(self, p):
        '''instruction : NOISE chunks end'''

        # parse noise node
        self.parse_noise_output(p[2])

    def p_noisy(self, p):
        '''instruction : NOISY chunks end'''

        # parse noisy node
        self.parse_noisy_source(p[2])

    def p_chunks(self, p):
        '''chunks : CHUNK
                  | chunks CHUNK'''
        p[0] = p[1]

        if len(p) == 3:
            p[0] += " " + p[2]

    def p_error(self, p):
        lineno = self.lineno

        if p:
            if hasattr(p, 'value'):
                # parser object
                
                # check for unexpected new line
                if p.value == "\n":
                    message = "unexpected end of line"
                    # compensate for mistaken newline
                    lineno -= 1
                else:
                    message = "'%s'" % p.value
            else:
                # error message thrown by production
                message = str(p)

                # productions always end with newlines, so errors in productions are on previous lines
                lineno -= 1
        else:
            message = "unexpected end of file"
        
        raise LisoParserError(message, lineno)

    def parse_passive(self, passive_type, *params):
        if len(params) != 4:
            self.p_error("unexpected parameter count (%d)" % len(params))

        arg_names = ["name", "value", "node1", "node2"]
        kwargs = {name: value for name, value in zip(arg_names, params)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        self.p_error("unrecognised passive component '{cmp}'".format(cmp=passive_type))

    def parse_library_opamp(self, *params):
        if len(params) < 5 or len(params) > 6:
            self.p_error("unexpected parameter count (%d)" % len(params))
        
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
                self.p_error("op-amp parameter override must be in the form 'param=value'")
            
            if key not in self.OP_OVERRIDE_MAP.keys():
                self.p_error("unknown op-amp override parameter '{key}'".format(key=key))
            
            extra_args[self.OP_OVERRIDE_MAP[key]] = value
        
        return extra_args

    def parse_frequencies(self, *params):
        if len(params) != 4:
            self.p_error("unexpected parameter count (%d)" % len(params))
        
        scale = params[0]
        start = Quantity(params[1], "Hz")
        stop = Quantity(params[2], "Hz")
        # LISO simulates specified steps + 1
        count = int(params[3]) + 1

        if scale.lower() == "lin":
            self.frequencies = np.linspace(start, stop, count)
        elif scale.lower() == "log":
            self.frequencies = np.logspace(np.log10(start), np.log10(stop),
                                           count)
        else:
            self.p_error("invalid frequency scale '{scale}'".format(scale=scale))

    def parse_voltage_input(self, *params):
        if len(params) < 1 or len(params) > 3:
            self.p_error("unexpected parameter count (%d)" % len(params))
        
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
            self.p_error("unexpected parameter count (%d)" % len(params))
        
        self.input_type = "current"

        # only a positive node
        self.input_node_p = params[0]

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
                    # all op-amp noises
                    self._source_all_opamps = True
                elif component_name.lower() == "allr":
                    # all resistor noises
                    self._source_all_resistors = True
                elif component_name.lower() == "sum":
                    # sum of circuit noises
                    self._source_sum = True
                else:
                    # individual component
                    definition = [component_name]
                    
                    if len(source_pieces) > 1:
                        # add op-amp noise type
                        definition.append(source_pieces[1])
                    
                    self._noise_source_defs.append(definition)

    def parse_noisy_source(self, noisy_str):
        """Set contributions to "sum" noise curve"""
        # split by whitespace
        params = noisy_str.split()

        # parse noisy sources
        for source_str in params:
            # strip any remaining whitespace
            source_str = source_str.strip()

            # split off op-amp port settings
            source_pieces = source_str.split(":")

            component_name = source_pieces[0]

            if component_name.lower() == "all":
                # all components noisy
                self._noisy_all_components = True
            elif component_name.lower() == "allop":
                self._noisy_all_opamps = True
            elif component_name.lower() == "allr":
                # all resistor noises
                self._noisy_all_resistors = True
            else:
                # individual component
                definition = [component_name]
                
                if len(source_pieces) > 1:
                    # add op-amp noise type
                    definition.append(source_pieces[1])
                
                self._noisy_source_defs.append(definition)