"""Base LISO parser"""

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

    def parse(self, filepath=None, text=None):
        if filepath is None and text is None:
            raise ValueError("must provide either a LISO filepath or LISO text")
        elif filepath is not None and text is not None:
            raise ValueError("cannot both parse from file and text")

        if filepath is not None:            
            with open(filepath, "r") as obj:
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
        return [element.element for element in self.tf_outputs if isinstance(element.element, Node)]

    @property
    def output_components(self):
        return [element.element for element in self.tf_outputs if isinstance(element.element, Component)]

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

    def __init__(self, element, scales, index=None, output_type=None):
        self.element = element
        self.scales = scales

        if index is not None:
            self.index = int(index)

        if output_type is not None:
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
    def __init__(self, *args, **kwargs):
        super(LisoOutputVoltage, self).__init__(output_type="voltage", *args, **kwargs)

class LisoOutputCurrent(LisoOutputElement):
    def __init__(self, *args, **kwargs):
        super(LisoOutputCurrent, self).__init__(output_type="current", *args, **kwargs)

class LisoNoiseSource(object):
    def __init__(self, noise, index=None, flags=None):
        self.noise = noise

        if index is not None:
            self.index = index
        
        if flags is not None:
            self.flags = flags