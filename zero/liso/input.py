"""LISO input file parser"""

import logging
import numpy as np

from ..format import Quantity
from ..components import OpAmp, NoiseNotFoundError
from .base import LisoParser, LisoOutputVoltage, LisoOutputCurrent, LisoParserError

LOGGER = logging.getLogger(__name__)

class LisoInputParser(LisoParser):
    """LISO input file parser

    This implements a lexer to identify appropriate definitions in a LISO input file,
    and a parser to build a circuit from what is found.
    """
    # dict mapping LISO op-amp parameter overrides to `zero.components.OpAmp` arguments
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
        "m": "M",
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
        self._output_all_nodes_scales = None
        self._output_all_opamp_nodes = False
        self._output_all_opamp_nodes_scales = None
        self._output_all_components = False
        self._output_all_components_scales = None
        self._output_all_opamps = False
        self._output_all_opamps_scales = None

        # raw noise source lists
        self._noise_defs = [] # displayed noise
        self._noisy_extra_defs = [] # extra noise to include in "sum" in addition to displayed noise

        # noise sum request
        self._noise_sum_to_be_computed = False

        super().__init__(*args, **kwargs)

    def _do_build(self):
        super()._do_build()

        # add extra node outputs
        if self._output_all_nodes:
            # show all nodes
            for node in self.circuit.non_gnd_nodes:
                if node.name in self.output_nodes:
                    # already present
                    continue

                sink = LisoOutputVoltage(node=node.name, scales=self._output_all_nodes_scales)
                self.add_tf_output(sink)
        elif self._output_all_opamp_nodes:
            # show all op-amp nodes
            for node in self.circuit.opamp_output_nodes:
                if node.name in self.output_nodes:
                    # already present
                    continue

                sink = LisoOutputVoltage(node=node.name, scales=self._output_all_opamp_nodes_scales)
                self.add_tf_output(sink)

        # add extra component outputs
        if self._output_all_components:
            # show all components
            for component in self.circuit.components:
                if component.name in self.output_components:
                    # already present
                    continue

                sink = LisoOutputCurrent(component=component.name,
                                         scales=self._output_all_components_scales)
                self.add_tf_output(sink)
        elif self._output_all_opamps:
            # show all op-amps
            for component in self.circuit.opamps:
                if component.name in self.output_components:
                    # already present
                    continue

                sink = LisoOutputCurrent(component=component.name,
                                         scales=self._output_all_opamps_scales)
                self.add_tf_output(sink)

    def _get_noise_objects(self, definitions):
        """Get noise objects for the specified raw noise defintions"""
        sources = set()

        for definition in definitions:
            if len(definition) > 2:
                self.p_error("unexpected extra ':'")

            component_name = definition[0].lower()

            if len(definition) > 1:
                suffix = definition[1]
            else:
                # no suffices specified
                suffix = None

            if component_name == "all":
                # show all noise sources
                for component in self.circuit.components:
                    if isinstance(component, OpAmp):
                        sources.update(self._get_component_noise(component, suffix))
                    else:
                        sources.update(self._get_component_noise(component))
            elif component_name == "allop":
                # show all op-amp noise sources
                for opamp in self.circuit.opamps:
                    sources.update(self._get_component_noise(opamp, suffix))
            elif component_name == "allr":
                # show all resistor noise sources
                if suffix is not None:
                    self.p_error("cannot specify suffix '%s' for 'allr'" % suffix)

                for resistor in self.circuit.resistors:
                    sources.update(self._get_component_noise(resistor))
            elif component_name == "sum":
                # show sum of circuit noises
                self._noise_sum_to_be_computed = True
            else:
                # individual component
                component = self.circuit[component_name]
                sources.update(self._get_component_noise(component, suffix))

        return sources

    def _get_component_noise(self, component, suffix=None):
        """Get noise list from specified component given the optionally specified suffix"""
        if suffix is not None:
            if not isinstance(component, OpAmp):
                self.p_error("noise suffices cannot be specified on non-op-amps")

            noise = []

            for character in suffix:
                lchar = character.lower()

                if lchar == "u":
                    # voltage noise source
                    try:
                        noise.append(component.voltage_noise)
                    except NoiseNotFoundError:
                        # noise not found
                        pass
                elif lchar == "+":
                    # non-inverting input current noise
                    try:
                        noise.append(component.non_inv_current_noise)
                    except NoiseNotFoundError:
                        # noise not found
                        pass
                elif lchar == "-":
                    # inverting input current noise
                    try:
                        noise.append(component.inv_current_noise)
                    except NoiseNotFoundError:
                        # noise not found
                        pass
                else:
                    self.p_error("unrecognised op-amp noise suffix '%s'" % character)
        else:
            # all noise sources
            noise = component.noise

        return noise

    @property
    def displayed_noise_objects(self):
        """Noise sources to be plotted"""
        return self._get_noise_objects(self._noise_defs)

    @property
    def summed_noise_objects(self):
        """Noise sources included in the sum column"""
        # check "sum" is not present in noisy definitions
        if "sum" in [definition[0].split(":")[0] for definition in self._noisy_extra_defs]:
            self.p_error("cannot specify 'sum' as noisy source")

        sum_sources = self._get_noise_objects(self._noisy_extra_defs)

        # add displayed noise sources if not already present
        sum_sources.update(self.displayed_noise_objects)

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
        # allow "-" only after first character, for parsing e.g. frequencies but not
        # negative numbers
        r'[a-zA-Z0-9_=.:][a-zA-Z0-9_=.:+\-]*'
        # check if chunk is a keyword
        t.type = self.reserved.get(t.value.lower(), 'CHUNK')
        return t

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
        self._parse_passive("r", *p[2:-1])

    def p_capacitor(self, p):
        '''instruction : C CHUNK CHUNK CHUNK CHUNK end'''
        # parse as capacitor
        self._parse_passive("c", *p[2:-1])

    def p_inductor(self, p):
        '''instruction : L CHUNK CHUNK CHUNK CHUNK end'''
        # parse as inductor
        self._parse_passive("l", *p[2:-1])

    def p_mutual_inductance(self, p):
        '''instruction : M CHUNK CHUNK CHUNK CHUNK end'''
        # parse as mutual inductance
        self._parse_mutual_inductance(*p[2:-1])

    def p_opamp(self, p):
        '''instruction : OP CHUNK CHUNK CHUNK CHUNK CHUNK end
                       | OP CHUNK CHUNK CHUNK CHUNK CHUNK chunks end'''
        # parse as op-amp
        self._parse_library_opamp(*p[2:-1])

    def p_freq(self, p):
        '''instruction : FREQ CHUNK CHUNK CHUNK CHUNK end'''
        # parse frequencies
        self._parse_frequencies(*p[2:-1])

    def p_uinput(self, p):
        '''instruction : UINPUT CHUNK end
                       | UINPUT CHUNK CHUNK end
                       | UINPUT CHUNK CHUNK CHUNK end'''
        # parse voltage input
        self._parse_voltage_input(*p[2:-1])

    def p_iinput(self, p):
        '''instruction : IINPUT CHUNK end
                       | IINPUT CHUNK CHUNK end'''
        # parse current input
        self._parse_current_input(*p[2:-1])

    def p_uoutput(self, p):
        '''instruction : UOUTPUT chunks end'''
        # parse voltage outputs
        self._parse_voltage_output(p[2])

    def p_ioutput(self, p):
        '''instruction : IOUTPUT chunks end'''
        # parse current outputs
        self._parse_current_output(p[2])

    def p_noise(self, p):
        '''instruction : NOISE CHUNK chunks end'''
        noise_str = p[2] + " " + p[3]

        # parse noise node
        self._parse_noise_output(noise_str)

    def p_noisy(self, p):
        '''instruction : NOISY chunks end'''
        # parse noisy node
        self._parse_noisy_source(p[2])

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
                # check for unexpected new line or end of file
                if p.type == "EOF":
                    message = "unexpected end of file"
                    # compensate for mistaken newline
                    lineno -= 1
                elif p.value.startswith("\n"):
                    message = "unexpected end of line"
                    # compensate for mistaken newlines
                    lineno -= p.value.count("\n")
                else:
                    message = "'%s'" % p.value
            else:
                # error message thrown by production
                message = str(p)

                # productions always end with newlines, so errors in productions are on previous
                # lines
                if lineno is not None:
                    lineno -= 1
        else:
            message = "unexpected end of file"

        raise LisoParserError(message, line=lineno)

    def _parse_passive(self, passive_type, *params):
        if len(params) != 4:
            self.p_error("unexpected parameter count (%d)" % len(params))

        arg_names = ["name", "value", "node1", "node2"]
        kwargs = {name: value for name, value in zip(arg_names, params)}

        if passive_type == "r":
            self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            self.circuit.add_inductor(**kwargs)
        else:
            self.p_error("unrecognised passive component '{cmp}'".format(cmp=passive_type))

    def _parse_mutual_inductance(self, name, coupling_factor, inductor_1, inductor_2):
        self._inductor_couplings.append((name, coupling_factor, inductor_1, inductor_2))

    def _parse_library_opamp(self, *params):
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

    def _parse_frequencies(self, *params):
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

    def _parse_voltage_input(self, *params):
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

    def _parse_current_input(self, *params):
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

    def _parse_voltage_output(self, output_str):
        # transfer function output
        self.output_type = "tf"

        for param in output_str.split():
            # split option by colon
            self._add_voltage_output(param)

    def _add_voltage_output(self, param_str):
        params = param_str.split(":")
        node_name = params[0]
        scales = params[1:]

        if node_name.lower() == "all":
            # all node voltages
            self._output_all_nodes = True
            self._output_all_nodes_scales = scales
        elif node_name.lower() == "allop":
            # op-amp outputs voltages
            self._output_all_opamp_nodes = True
            self._output_all_opamp_nodes_scales = scales
        else:
            # add output
            self.add_tf_output(LisoOutputVoltage(node=node_name, scales=scales))

    def _parse_current_output(self, output_str):
        # transfer function output
        self.output_type = "tf"

        for param in output_str.split():
            # split option by colon
            self._add_current_output(param)

    def _add_current_output(self, param_str):
        params = param_str.split(":")
        component_name = params[0]
        scales = params[1:]

        if component_name.lower() == "all":
            # all component currents
            self._output_all_components = True
            self._output_all_components_scales = scales
        elif component_name.lower() == "allop":
            # op-amp output currents
            self._output_all_opamps = True
            self._output_all_opamps_scales = scales
        else:
            # add output
            self.add_tf_output(LisoOutputCurrent(component=component_name, scales=scales))

    def _parse_noise_output(self, noise_str):
        # split by whitespace
        params = noise_str.split()

        # noise output
        self.output_type = "noise"

        # noise output
        self.noise_output_node = params[0]

        # parse noise sources
        for source_str in params[1:]:
            # strip any remaining whitespace
            source_str = source_str.strip()

            # split off op-amp port settings
            source_pieces = source_str.split(":")

            if len(source_pieces) > 2:
                # too many colons
                self.p_error("unexpected extra ':'")

            # add raw noise definition and any suffices
            self._noise_defs.append(source_pieces)

    def _parse_noisy_source(self, noisy_str):
        """Set contributions to "sum" noise curve"""
        # split by whitespace
        params = noisy_str.split()

        # parse noisy sources
        for source_str in params:
            # strip any remaining whitespace
            source_str = source_str.strip()

            # split off op-amp port settings
            source_pieces = source_str.split(":")

            # add raw noise definition and any suffices
            self._noisy_extra_defs.append(source_pieces)
