import numpy as np
import logging
import abc
from ply import lex, yacc
from ply.lex import TOKEN

from circuit.circuit import Circuit
from circuit.format import SIFormatter
from circuit.solution import Solution
from circuit.analysis.ac import SmallSignalAcAnalysis

LOGGER = logging.getLogger("liso")

class LisoInputFormatException(Exception):
    pass

class LisoParser(object, metaclass=abc.ABCMeta):
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

    def __init__(self, **kwargs):
        # initial line number
        self.lineno = 1

        # create circuit
        self.circuit = Circuit()

        # default circuit values
        self.frequencies = None
        self.input_type = None
        self.input_node_n = None
        self.input_node_p = None
        self.input_impedance = None
        self._output_type = None
        self.output_nodes = set()
        self.output_components = set()
        self.output_all_nodes = False
        self.output_all_opamp_nodes = False
        self.output_all_components = False
        self.output_all_opamp_components = False
        self.noise_node = None

        # circuit solution
        self._solution = None

        # create lexer and parser handlers
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self)

        # parse the file or string
        self.parse(**kwargs)

    def parse(self, filepath=None, text=None):
        if filepath is None and text is None:
            raise ValueError("must provide either a LISO filepath or LISO text")
        
        if filepath is not None:
            with open(filepath, "r") as obj:
                text = obj.readlines()

        self.parser.parse(text, lexer=self.lexer)

        # check the parsed text is valid
        self.validate()

    def show(self, *args, **kwargs):
        """Show LISO results"""

        if not self.plottable:
            LOGGER.warning("nothing to show")

        # get solution
        solution = self.solution(*args, **kwargs)

        # draw plots
        solution.plot(output_nodes=self.output_nodes,
                      output_components=self.output_components)
        # display plots
        solution.show()

    def run(self, *args, **kwargs):
        # add input component, if not yet present
        self._set_circuit_input()

        if self.output_type == "tf":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_tfs(
                frequencies=self.frequencies,
                output_components=self.output_components,
                output_nodes=self.output_nodes,
                *args, **kwargs)
        elif self.output_type == "noise":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_noise(
                frequencies=self.frequencies,
                noise_node=self.noise_node,
                *args, **kwargs)
        
        raise SyntaxError("no outputs requested")

    def solution(self, force=False, *args, **kwargs):
        if not self._solution or force:
            self._solution = self.run(*args, **kwargs)
        
        return self._solution

    def validate(self):
        if self.frequencies is None:
            # no frequencies found
            raise SyntaxError("no plot frequencies found")
        elif (self.input_node_n is None and self.input_node_p is None):
            # no input nodes found
            raise SyntaxError("no input nodes found")
        elif ((len(self.output_nodes) == 0 and not self.output_all_nodes)
              and (len(self.output_components) == 0 and not self.output_all_components)
              and self.noise_node is None):
            # no output requested
            raise SyntaxError("no output requested")

    @property
    def will_calc_tfs(self):
        return self.will_calc_node_tfs or self.will_calc_component_tfs

    @property
    def will_calc_node_tfs(self):
        return len(self.output_nodes) > 0

    @property
    def will_calc_component_tfs(self):
        return len(self.output_components) > 0

    @property
    def will_calc_noise(self):
        return self.noise_node is not None

    @property
    def plottable(self):
        return self.will_calc_tfs or self.will_calc_noise

    @property
    def output_nodes(self):
        if self.output_all_nodes:
            return set(self.circuit.non_gnd_nodes)

        # requested output nodes
        output_nodes = self._output_nodes

        # add op-amps if necessary
        if self.output_all_opamp_nodes:
            output_nodes |= self.opamp_output_node_names

        return output_nodes

    @output_nodes.setter
    def output_nodes(self, nodes):
        self._output_nodes = set(nodes)

        # unset all outputs flags
        self.output_all_nodes = False
        self.output_all_opamp_nodes = False

    @property
    def output_components(self):
        if self.output_all_components:
            return set(self.circuit.components)

        # requested output components
        output_components = self._output_components

        # add op-amps if necessary
        if self.output_all_opamp_components:
            output_components |= self.opamp_names

        return output_components

    @output_components.setter
    def output_components(self, components):
        self._output_components = set(components)

        # unset all outputs flags
        self.output_all_components = False
        self.output_all_opamp_components = False

    @property
    def opamp_output_node_names(self):
        """Get set of node names associated with outputs of opamps in the \
           circuit"""
        return set([node.name for node in [opamp.node3 for
                                           opamp in self.circuit.opamps]])

    @property
    def opamp_names(self):
        """Get set of op-amp component names in the circuit"""
        return set([component.name for component in self.circuit.opamps])

    @property
    def output_type(self):
        return self._output_type
    
    @output_type.setter
    def output_type(self, output_type):
        if self.output_type is not None:
            if self.output_type != output_type:
                # output type changed
                raise Exception("output file contains both transfer functions "
                                "and noise, which is not supported")

            # output type isn't being changed; no need to do anything else
            return

        if output_type not in ["tf", "noise"]:
            raise ValueError("unknown output type")

        self._output_type = output_type

    def _set_circuit_input(self):
        # create input component if necessary
        try:
            self.circuit.get_component("input")
        except ValueError:
            # add input
            input_type = self.input_type
            node = None
            node_p = None
            node_n = None
            impedance = None

            if self.input_node_n is None:
                # fixed input
                node = self.input_node_p
            else:
                # floating input
                node_p = self.input_node_p
                node_n = self.input_node_n

            # input type depends on whether we calculate noise or transfer
            # functions
            if self.noise_node is not None:
                # we're calculating noise
                input_type = "noise"

                # set input impedance
                impedance = self.input_impedance

            self.circuit.add_input(input_type=input_type, node=node,
                                   node_p=node_p, node_n=node_n,
                                   impedance=impedance)


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


if __name__ == "__main__":
    parser = LisoInputParser(text="""
c c1 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout

freq log 1 100k 100

uinput nin 0
uoutput nout:db:deg

""")

    parser.show()