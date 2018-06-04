"""Base LISO parser"""

import os
import abc
from ply import lex, yacc
import logging
import numpy as np
from collections import defaultdict

from ..circuit import Circuit
from ..components import Component, Node
from ..analysis.ac import SmallSignalAcAnalysis
from ..format import SIFormatter

LOGGER = logging.getLogger("liso")

class LisoParser(object, metaclass=abc.ABCMeta):
    def __init__(self):
        # initial line number
        self.lineno = 1
        self._previous_newline_position = 0

        # create circuit
        self.circuit = Circuit()

        # circuit built status
        self.circuit_built = False

        # default circuit values
        self.frequencies = None
        self.input_type = None
        self.input_node_n = None
        self.input_node_p = None
        self.input_impedance = None
        self._output_type = None
        self.tf_outputs = []
        self.output_all_nodes = False
        self.output_all_opamp_nodes = False
        self.output_all_components = False
        self.output_all_opamp_components = False
        self.noise_output_node = None
        self.noise_sources = []

        # circuit solution
        self._solution = None

        # create lexer and parser handlers
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self)

    def parse(self, text):
        if text is None:
            raise ValueError("must provide either a filepath or text")

        if os.path.isfile(text):
            with open(text, "r") as obj:
                text = obj.read()

        #self.lexer.input(text)
        # Tokenize
        #while True:
        #    tok = self.lexer.token()
        #    if not tok: 
        #        break      # No more input
        #    print(tok)

        # add newline to end of text
        # this allows parsers to use newline characters to separate lines
        text += "\n"

        self.parser.parse(text, lexer=self.lexer)

    def show(self, *args, **kwargs):
        """Show LISO results"""

        # build circuit if necessary
        self.build()

        if not self.plottable:
            LOGGER.warning("nothing to show")

        # get solution
        solution = self.solution(*args, **kwargs)

        # draw plots
        solution.plot(output_nodes=self.output_nodes,
                      output_components=self.output_components)
        # display plots
        solution.show()

    def solution(self, force=False, *args, **kwargs):
        # build circuit if necessary
        self.build()
        
        if not self._solution or force:
            self._solution = self.run(*args, **kwargs)
        
        return self._solution

    def run(self, *args, **kwargs):
        # build circuit if necessary
        self.build()

        if self.output_type == "tf":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_tfs(
                frequencies=self.frequencies,
                output_components=self.output_components,
                output_nodes=self.output_nodes,
                *args, **kwargs)
        elif self.output_type == "noise":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_noise(
                frequencies=self.frequencies,
                noise_node=self.noise_output_node,
                *args, **kwargs)
        
        raise SyntaxError("no outputs requested")

    def build(self):
        """Build circuit if not yet built"""

        if not self.circuit_built:
            self._do_build()

            # check the circuit is valid
            self.validate()

            # set built flag
            self.circuit_built = True

    def _do_build(self):
        """Build circuit"""

        # add input component, if not yet present
        self._set_circuit_input()

    def validate(self):
        if self.frequencies is None:
            # no frequencies found
            raise SyntaxError("no plot frequencies found")
        elif (self.input_node_n is None and self.input_node_p is None):
            # no input nodes found
            raise SyntaxError("no input nodes found")
        elif (len(self.tf_outputs) == 0 and self.noise_output_node is None):
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
        return self.noise_output_node is not None

    @property
    def plottable(self):
        return self.will_calc_tfs or self.will_calc_noise

    @property
    def output_nodes(self):
        nodes = set([element.element for element in self.tf_outputs if element.type == "node"])

        if "all" in nodes:
            nodes.update([node.name for node in self.circuit.non_gnd_nodes])
        elif "allop" in nodes:
            nodes.update(self.opamp_output_node_names)
        
        # remove special keywords
        nodes.difference_update(["all", "allop"])

        return nodes

    @property
    def output_components(self):
        components = set([element.element for element in self.tf_outputs if element.type == "component"])

        if "all" in components:
            components.update([component.name for component in self.circuit.components])
        elif "allop" in components:
            components.update(self.opamp_names)
        
        # remove special keywords
        components.difference_update(["all", "allop"])

        return components

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

            if self.input_node_p is None:
                pass#raise SyntaxError("no input node specified")

            if self.input_node_n is None:
                # fixed input
                node = self.input_node_p
            else:
                # floating input
                node_p = self.input_node_p
                node_n = self.input_node_n

            # input type depends on whether we calculate noise or transfer
            # functions
            if self.noise_output_node is not None:
                # we're calculating noise
                input_type = "noise"

                # set input impedance
                impedance = self.input_impedance

            self.circuit.add_input(input_type=input_type, node=node,
                                   node_p=node_p, node_n=node_n,
                                   impedance=impedance)

class LisoOutputElement(object):
    magnitude_scales = ["dB", "Abs"]
    phase_scales = ["Degrees", "Degrees (>0)", "Degrees (<0)", "Degrees (continuous)"]
    real_scales = ["Real"]
    imag_scales = ["Imag"]

    def __init__(self, _type, element=None, scales=[], index=None, output_type=None):
        scales = list(scales)
        
        if index is not None:
            index = int(index)

        self.type = str(_type)
        self.element = element
        self.scales = scales
        self.index = index
        self.output_type = output_type
    
    @property
    def n_scales(self):
        return len(self.scales)

    @property
    def has_magnitude(self):
        return any([scale in self.scales for scale in self.magnitude_scales])
    
    @property
    def has_phase(self):
        return any([scale in self.scales for scale in self.phase_scales])

    @property
    def has_real(self):
        return any([scale in self.scales for scale in self.real_scales])

    @property
    def has_imag(self):
        return any([scale in self.scales for scale in self.imag_scales])

    @property
    def magnitude_index(self):
        return self.get_scale(self.magnitude_scales)

    @property
    def phase_index(self):
        return self.get_scale(self.phase_scales)

    @property
    def real_index(self):
        return self.get_scale(self.real_scales)

    @property
    def imag_index(self):
        return self.get_scale(self.imag_scales)

    def get_scale(self, scale_names):
        for index, scale in enumerate(self.scales):
            if scale in scale_names:
                return index, scale

        raise ValueError("scale names not found")

class LisoOutputVoltage(LisoOutputElement):
    def __init__(self, node=None, all_nodes=False, all_opamps=False, *args, **kwargs):
        super().__init__(_type="node", element=node, output_type="voltage", *args, **kwargs)

        self.all_nodes = bool(all_nodes)
        self.all_opamps = bool(all_opamps)

class LisoOutputCurrent(LisoOutputElement):
    def __init__(self, component=None, all_components=False, all_opamps=False, *args, **kwargs):
        super().__init__(_type="component", element=component, output_type="current", *args, **kwargs)

        self.all_components = bool(all_components)
        self.all_opamps = bool(all_opamps)

class LisoNoiseSource(object):
    def __init__(self, noise, index=None, flags=None):
        self.noise = noise

        if index is not None:
            self.index = index
        
        if flags is not None:
            self.flags = flags