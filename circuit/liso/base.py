"""Base LISO parser"""

import abc
from ply import lex, yacc
import logging

from ..circuit import Circuit
from ..analysis.ac import SmallSignalAcAnalysis

LOGGER = logging.getLogger("liso")

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
                text = obj.read()

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