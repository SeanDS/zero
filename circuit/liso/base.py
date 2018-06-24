"""Base LISO parser"""

import sys
import os
import abc
import logging
from ply import lex, yacc
import numpy as np

from ..circuit import Circuit
from ..components import Node
from ..analysis import AcSignalAnalysis, AcNoiseAnalysis
from ..format import Quantity

LOGGER = logging.getLogger("liso")

class LisoParserError(ValueError):
    """LISO parser error"""
    def __init__(self, message, line=None, pos=None, **kwargs):
        if line is not None:
            line = int(line)

            if pos is not None:
                pos = int(pos)

                # add line number and position
                message = "{message} (line {line}, position {pos})".format(message=message,
                                                                           line=line, pos=pos)
            else:
                # add line number
                message = "{message} (line {line})".format(message=message, line=line)

        # prepend message
        message = "LISO syntax error: {message}".format(message=message)

        super().__init__(message, **kwargs)

class LisoParser(object, metaclass=abc.ABCMeta):
    """Base LISO parser"""
    def __init__(self):
        # initial line number
        self.lineno = 1
        self._previous_newline_position = 0

        # create circuit
        self.circuit = Circuit()

        # circuit built status
        self._circuit_built = False

        # default circuit values
        self._frequencies = None
        self._input_type = None
        self._input_node_n = None
        self._input_node_p = None
        self._input_impedance = None
        self._output_type = None
        self.tf_sinks = []

        # the node noise is projected to
        self._noise_output_node = None

        # noise sources to calculate
        self._noise_sources = None

        # noise sum flags
        self._noise_sum_requested = False # a noise sum should be computed
        self._noise_sum_present = False   # a noise sum is present in the data

        # circuit solution
        self._solution = None

        # create lexer and parser handlers
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self)

        # whether parser end of file has been reached
        self._eof = False

    @property
    def frequencies(self):
        """Analysis frequencies"""
        return self._frequencies

    @frequencies.setter
    def frequencies(self, frequencies):
        if self._frequencies is not None:
            self.p_error("cannot redefine frequencies")

        self._frequencies = np.array(frequencies)

    @property
    def input_type(self):
        """Circuit input type"""
        return self._input_type

    @input_type.setter
    def input_type(self, input_type):
        if self._input_type is not None:
            self.p_error("cannot redefine input type")

        self._input_type = input_type

    @property
    def input_node_p(self):
        """Circuit positive input node"""
        return self._input_node_p

    @input_node_p.setter
    def input_node_p(self, input_node_p):
        if self._input_node_p is not None:
            self.p_error("cannot redefine positive input node")

        self._input_node_p = Node(input_node_p)

    @property
    def input_node_n(self):
        """Circuit negative input node"""
        return self._input_node_n

    @input_node_n.setter
    def input_node_n(self, input_node_n):
        if self._input_node_n is not None:
            self.p_error("cannot redefine negative input node")

        self._input_node_n = Node(input_node_n)

    @property
    def input_impedance(self):
        """Circuit input impedance"""
        return self._input_impedance

    @input_impedance.setter
    def input_impedance(self, input_impedance):
        if self._input_impedance is not None:
            self.p_error("cannot redefine input impedance")

        self._input_impedance = Quantity(input_impedance, "Ω")

    @property
    def noise_output_node(self):
        """Node that noise sources in the circuit are projected to"""
        return self._noise_output_node

    @noise_output_node.setter
    def noise_output_node(self, noise_output_node):
        if self._noise_output_node is not None:
            self.p_error("cannot redefine noise output node")

        self._noise_output_node = Node(noise_output_node)

    def add_tf_sink(self, sink):
        """Add transfer function sink.

        This stores the specified sink for use in building the solution.

        Parameters
        ----------
        sink : :class:`LisoOutputElement`
            The sink to add.

        Raises
        ------
        :class:`ValueError`
            If the specified sink is already present.
        """
        if sink in self.tf_sinks:
            raise ValueError("sink '%s' is already present" % sink)

        self.tf_sinks.append(sink)

    def parse(self, text=None, path=None):
        if text is None and path is None:
            raise ValueError("must provide either text or a path")

        if path is not None:
            if text is not None:
                raise ValueError("cannot specify both text and a file to parse")

            if not os.path.isfile(path):
                raise FileNotFoundError("cannot read '{path}'".format(path=path))

            with open(path, "r") as obj:
                text = obj.read()

        if self._eof:
            # reset end of file
            self._eof = False

        self.parser.parse(text, lexer=self.lexer)

    @abc.abstractmethod
    def p_error(self, p):
        """Child classes must implement error handler"""
        raise NotImplementedError

    def show(self, *args, **kwargs):
        """Show LISO results"""
        # build circuit if necessary
        self.build()

        if not self.plottable:
            LOGGER.warning("nothing to show")

        # get solution
        solution = self.solution(*args, **kwargs)

        # draw plots
        if self.output_type == "tf":
            solution.plot_tfs(sources=self.default_tf_sources(), sinks=self.default_tf_sinks())
        elif self.output_type == "noise":
            sum_kwargs = {}

            if self._noise_sum_requested:
                # calculate and plot noise sum
                sum_kwargs["compute_sum_sources"] = self.summed_noise_sources

            solution.plot_noise(sources=self.displayed_noise_sources,
                                show_sums=self._noise_sum_present, **sum_kwargs)
        else:
            raise Exception("unrecognised output type")

        # display plots
        solution.show()

    def default_tf_sources(self):
        """Default transfer function sources"""
        # note: this cannot be a property, otherwise lex will execute the property before
        # input_type is set.
        if self.input_type == "voltage":
            sources = [self.circuit.input_component.node2]
        elif self.input_type == "current":
            sources = [self.circuit.input_component]
        else:
            raise ValueError("unrecognised input type")

        return sources

    def default_tf_sinks(self):
        """Default transfer function sinks."""
        return self.tf_sinks

    def solution(self, force=False, **kwargs):
        """Get the solution to the analysis defined in the parsed file"""
        # build circuit if necessary
        self.build()

        if not self._solution or force:
            self._solution = self._run(**kwargs)

        return self._solution

    def _run(self, print_equations=False, print_matrix=False, stream=sys.stdout):
        # build circuit if necessary
        self.build()

        if self.output_type == "tf":
            analysis = AcSignalAnalysis(circuit=self.circuit, frequencies=self.frequencies)
        elif self.output_type == "noise":
            analysis = AcNoiseAnalysis(circuit=self.circuit, frequencies=self.frequencies,
                                       node=self.noise_output_node)
        else:
            raise SyntaxError("no outputs requested")

        if print_equations:
            print(analysis.circuit_equation_display(), file=stream)

        if print_matrix:
            print(analysis.circuit_matrix_display(), file=stream)

        analysis.calculate()

        return analysis.solution

    def build(self):
        """Build circuit if not yet built"""

        if not self._circuit_built:
            self._do_build()

            # check the circuit is valid
            self._validate()

            # set built flag
            self._circuit_built = True

    def _do_build(self):
        """Build circuit"""

        # unset lineno to avoid using the last line in subsequent errors
        self.lineno = None

        # add input component, if not yet present
        self._set_circuit_input()

    def _validate(self):
        if self.frequencies is None:
            # no frequencies found
            raise SyntaxError("no plot frequencies found")
        elif self.input_node_n is None and self.input_node_p is None:
            # no input nodes found
            raise SyntaxError("no input nodes found")
        elif not self.tf_sinks and self.noise_output_node is None:
            # no output requested
            raise SyntaxError("no output requested")

    @property
    def will_calc_tfs(self):
        """Whether the analysis will calculate transfer functions"""
        return self.will_calc_node_tfs or self.will_calc_component_tfs

    @property
    def will_calc_node_tfs(self):
        """Whether the analysis will calculate transfer functions to nodes"""
        return len(self.output_nodes) > 0

    @property
    def will_calc_component_tfs(self):
        """Whether the analysis will calculate transfer functions to components"""
        return len(self.output_components) > 0

    @property
    def will_calc_noise(self):
        """Whether the analysis will calculate noise"""
        return self.noise_output_node is not None

    @property
    def plottable(self):
        """Whether the analysis will calculate something that can be plotted"""
        return self.will_calc_tfs or self.will_calc_noise

    @property
    def output_nodes(self):
        """The output nodes of the transfer functions computed in the analysis"""
        return set([element.element for element in self.tf_sinks if element.type == "node"])

    @property
    def output_components(self):
        """The output components of the transfer functions computed in the analysis"""
        return set([element.element for element in self.tf_sinks if element.type == "component"])

    @property
    @abc.abstractmethod
    def displayed_noise_sources(self):
        """Displayed noise sources, not including sums"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summed_noise_sources(self):
        """Noise sources included in displayed noise sum outputs"""
        raise NotImplementedError

    @property
    def opamp_output_node_names(self):
        """Get set of node names associated with outputs of opamps in the circuit"""
        return set([node.name for node in [opamp.node3 for
                                           opamp in self.circuit.opamps]])

    @property
    def opamp_names(self):
        """Get set of op-amp component names in the circuit"""
        return set([component.name for component in self.circuit.opamps])

    @property
    def resistor_names(self):
        """Get set of resistor component names in the circuit"""
        return set([component.name for component in self.circuit.resistors])

    @property
    def output_type(self):
        """Transfer function output type"""
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

    def _get_noise_sources(self, definitions):
        """Get noise objects for the specified raw noise defintions"""
        sources = []

        # create noise source objects
        for definition in definitions:
            component = self.circuit.get_component(definition[0])

            if len(definition) > 1:
                # op-amp noise type specified
                type_str = definition[1].lower()

                if type_str in ["u", "0"]:
                    noise = component.voltage_noise
                elif type_str in ["+", "i+", "1"]:
                    # non-inverting input current noise
                    noise = component.non_inv_current_noise
                elif type_str in ["-", "i-", "2"]:
                    # inverting input current noise
                    noise = component.inv_current_noise
                else:
                    self.p_error("unrecognised op-amp noise source '%s'" % definition[1])

                # add noise source
                sources.append(noise)
            else:
                # get all of the component's noise sources
                sources.extend(component.noise)

        return sources

class LisoOutputElement(object, metaclass=abc.ABCMeta):
    """LISO output element"""
    MAGNITUDE_SCALES = ["dB", "Abs"]
    PHASE_SCALES = ["Degrees", "Degrees (>0)", "Degrees (<0)", "Degrees (continuous)"]
    REAL_SCALES = ["Real"]
    IMAG_SCALES = ["Imag"]

    def __init__(self, type_, element=None, scales=None, index=None, output_type=None):
        if scales is None:
            scales = []

        if index is not None:
            index = int(index)

        self.type = str(type_)
        self.element = element
        self.scales = list(scales)
        self.index = index
        self.output_type = output_type

    def _get_scale(self, scale_names):
        for index, scale in enumerate(self.scales):
            if scale in scale_names:
                return index, scale

        raise ValueError("scale names not found")

    @property
    def n_scales(self):
        """Number of scales"""
        return len(self.scales)

    @property
    def has_magnitude(self):
        """Whether the output has a magnitude scale"""
        return any([scale in self.scales for scale in self.MAGNITUDE_SCALES])

    @property
    def has_phase(self):
        """Whether the output has a phase scale"""
        return any([scale in self.scales for scale in self.PHASE_SCALES])

    @property
    def has_real(self):
        """Whether the output has a real scale"""
        return any([scale in self.scales for scale in self.REAL_SCALES])

    @property
    def has_imag(self):
        """Whether the output has a imaginary scale"""
        return any([scale in self.scales for scale in self.IMAG_SCALES])

    @property
    def magnitude_index(self):
        """The index of the magnitude scale in the scale list"""
        return self._get_scale(self.MAGNITUDE_SCALES)

    @property
    def phase_index(self):
        """The index of the phase scale in the scale list"""
        return self._get_scale(self.PHASE_SCALES)

    @property
    def real_index(self):
        """The index of the real scale in the scale list"""
        return self._get_scale(self.REAL_SCALES)

    @property
    def imag_index(self):
        """The index of the imaginary scale in the scale list"""
        return self._get_scale(self.IMAG_SCALES)

class LisoOutputVoltage(LisoOutputElement):
    """LISO output voltage"""
    def __init__(self, *args, node=None, **kwargs):
        super().__init__(type_="node", element=node, output_type="voltage", *args, **kwargs)

    @property
    def node(self):
        """The output voltage node"""
        return self.element

class LisoOutputCurrent(LisoOutputElement):
    """LISO output current"""
    def __init__(self, *args, component=None, **kwargs):
        super().__init__(type_="component", element=component, output_type="current", *args,
                         **kwargs)

    @property
    def component(self):
        """The output current component"""
        return self.element

class LisoNoiseSource(object):
    """LISO noise source"""
    def __init__(self, noise, index=None):
        if index is not None:
            index = int(index)

        self.noise = noise
        self.index = index

    def __str__(self):
        return "LISO {noise}".format(noise=self.noise)
