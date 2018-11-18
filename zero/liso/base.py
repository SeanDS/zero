"""Base LISO parser"""

import sys
import os
import abc
import logging
from ply import lex, yacc
import numpy as np

from ..circuit import Circuit, ComponentNotFoundError
from ..components import Node
from ..analysis import AcSignalAnalysis, AcNoiseAnalysis
from ..data import MultiNoiseSpectrum
from ..format import Quantity

LOGGER = logging.getLogger(__name__)


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


class LisoParser(metaclass=abc.ABCMeta):
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
        self.tf_outputs = []

        # list of (name, coupling, l1, l2) inductor coupling tuples
        self._inductor_couplings = []

        # the node or component that circuit noise is projected to
        self._noise_output_element = None

        # noise sources to calculate
        self._noise_sources = None

        # circuit solution
        self._solution = None

        # flag for when noise sum must be computed when building solution
        self._noise_sum_to_be_computed = False

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
    def noise_output_element(self):
        """Node or component that noise sources in the circuit are projected to"""
        return self._noise_output_element

    @noise_output_element.setter
    def noise_output_element(self, noise_output_element):
        if self._noise_output_element is not None:
            self.p_error("cannot redefine noise output element")

        self._noise_output_element = Node(noise_output_element)

    def add_tf_output(self, output):
        """Add transfer function output.

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
        if output in self.tf_outputs:
            raise ValueError("sink '%s' is already present" % output)

        self.tf_outputs.append(output)

    @property
    def n_tf_outputs(self):
        return len(self.tf_outputs)

    @property
    def n_displayed_noise(self):
        return len(self.displayed_noise_objects)

    @property
    def n_summed_noise(self):
        return len(self.summed_noise_objects)

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

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        pos = t.lexer.lexpos - self._previous_newline_position

        raise LisoParserError("illegal character '{char}'".format(char=t.value[0]), self.lineno,
                              pos)

    @abc.abstractmethod
    def p_error(self, p):
        """Child classes must implement error handler"""
        raise NotImplementedError

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
        """Default transfer function sinks.

        Returns
        -------
        :class:`list` of :class:`Component` or :class:`Node`
        """
        return [element.element for element in self.tf_outputs]

    def solution(self, force=False, set_default_plots=True, **kwargs):
        """Get the solution to the analysis defined in the parsed file.

        Parameters
        ----------
        force : :class:`bool`
            Whether to force the solution to be recomputed if already generated.
        set_default_plots : :class:`bool`
            Set the plots defined in the LISO file as defaults.
        """
        # build circuit if necessary
        self.build()

        if not self._solution or force:
            self._solution = self._run(**kwargs)

            if self._noise_sum_to_be_computed:
                # find spectra in solution
                sum_spectra = self._solution.filter_noise(sources=self.summed_noise_objects)
                # create overall spectrum
                sum_spectrum = MultiNoiseSpectrum(sink=self.noise_output_element,
                                                  constituents=sum_spectra)
                # build noise sum and show by default
                self._solution.add_noise_sum(sum_spectrum, default=True)

        if set_default_plots:
            self._set_default_plots()

        return self._solution

    def _set_default_plots(self):
        # set default plots
        if self.output_type == "tf":
            default_tfs = self._solution.filter_tfs(sources=self.default_tf_sources(),
                                                    sinks=self.default_tf_sinks())

            for tf in default_tfs:
                if not self._solution.is_default_tf(tf):
                    self._solution.set_tf_as_default(tf)
        elif self.output_type == "noise":
            default_spectra = self._solution.filter_noise(sources=self.displayed_noise_objects)

            for spectrum in default_spectra:
                if not self._solution.is_default_noise(spectrum):
                    self._solution.set_noise_as_default(spectrum)

    def _run(self, print_equations=False, print_matrix=False, stream=sys.stdout, **kwargs):
        # build circuit if necessary
        self.build()

        if self.output_type == "tf":
            analysis = AcSignalAnalysis(circuit=self.circuit, frequencies=self.frequencies, **kwargs)
        elif self.output_type == "noise":
            analysis = AcNoiseAnalysis(circuit=self.circuit, frequencies=self.frequencies,
                                       node=self.noise_output_element, **kwargs)
        else:
            self.p_error("no outputs requested")

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

        # coupling between inductors
        self._set_inductor_couplings()

    def _validate(self):
        if self.frequencies is None:
            # no frequencies found
            self.p_error("no plot frequencies found")
        elif self.input_node_n is None and self.input_node_p is None:
            # no input nodes found
            self.p_error("no input nodes found")
        elif not self.tf_outputs and self.noise_output_element is None:
            # no output requested
            self.p_error("no output requested")

        # check for invalid output nodes
        for tf_output in self.tf_outputs:
            # get element
            element = tf_output.element

            if not self.circuit.has_component(element) and not self.circuit.has_node(element):
                self.p_error("output element '%s' is not present in the circuit" % element)

        # noise output element must exist
        if self.noise_output_element is not None:
            name = self.noise_output_element.name
            if not self.circuit.has_node(name):
                self.p_error("noise output element '%s' is not present in the circuit" % name)

        # check if noise sources exist
        try:
            _ = self.displayed_noise_objects
            _ = self.summed_noise_objects
        except ComponentNotFoundError as e:
            self.p_error("noise source '%s' is not present in the circuit" % e.name)

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
        return self.noise_output_element is not None

    @property
    def plottable(self):
        """Whether the analysis will calculate something that can be plotted"""
        return self.will_calc_tfs or self.will_calc_noise

    @property
    def output_nodes(self):
        """The output nodes of the transfer functions computed in the analysis"""
        return set([element.element for element in self.tf_outputs if element.type == "node"])

    @property
    def output_components(self):
        """The output components of the transfer functions computed in the analysis"""
        return set([element.element for element in self.tf_outputs if element.type == "component"])

    @property
    @abc.abstractmethod
    def displayed_noise_objects(self):
        """Displayed noise objects, not including sums"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def summed_noise_objects(self):
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
            self.circuit["input"]
        except ComponentNotFoundError:
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

            # input type depends on whether we calculate noise or transfer functions
            if self.noise_output_element is not None:
                # we're calculating noise
                input_type = "noise"

                # set input impedance
                impedance = self.input_impedance

            self.circuit.add_input(input_type=input_type, node=node,
                                   node_p=node_p, node_n=node_n,
                                   impedance=impedance)

    def _set_inductor_couplings(self):
        # discard name (not used in circuit mode)
        for _, value, inductor_1, inductor_2 in self._inductor_couplings:
            self.circuit.set_inductor_coupling(inductor_1, inductor_2, value)


class LisoOutputElement(metaclass=abc.ABCMeta):
    """LISO output element"""
    # supported scales
    SUPPORTED_SCALES = {"magnitude": {"db": ["db"], "abs": ["abs"]},
                        "phase": {"deg": ["deg", "degrees", "degrees (>0)", "degrees (<0)",
                                          "degrees (continuous)"]},
                        "real": {"real": ["re", "real"]},
                        "imaginary": {"imag": ["im", "imag"]}}

    def __init__(self, type_, element=None, scales=None, index=None, output_type=None):
        if scales is None:
            scales = []

        if index is not None:
            index = int(index)

        self._scales = None

        self.type = str(type_)
        self.element = element
        self.scales = scales
        self.index = index
        self.output_type = output_type

    @property
    def scales(self):
        return self._scales

    @scales.setter
    def scales(self, scales):
        scales_list = []
        for scale in scales:
            scales_list.append(self._parse_scale(scale))

        self._scales = scales_list

    def _parse_scale(self, raw_scale):
        """Identify specified scale"""
        candidate = raw_scale.lower()
        for scale_class in self.SUPPORTED_SCALES:
            for scale in self.SUPPORTED_SCALES[scale_class]:
                if candidate in self.SUPPORTED_SCALES[scale_class][scale]:
                    return scale

        raise ValueError("unrecognised scale: '%s'" % raw_scale)

    def _get_scale(self, scale_names):
        for index, scale in enumerate(self.scales):
            if scale in scale_names:
                return index, scale

        raise ValueError("scale not found in scale list")

    @property
    def n_scales(self):
        """Number of scales"""
        return len(self.scales)

    @property
    def has_magnitude(self):
        """Whether the output has a magnitude scale"""
        return any([scale in self.scales for scale in self.magnitude_scales])

    @property
    def has_phase(self):
        """Whether the output has a phase scale"""
        return any([scale in self.scales for scale in self.phase_scales])

    @property
    def has_real(self):
        """Whether the output has a real scale"""
        return any([scale in self.scales for scale in self.real_scales])

    @property
    def has_imag(self):
        """Whether the output has a imaginary scale"""
        return any([scale in self.scales for scale in self.imag_scales])

    @property
    def magnitude_index(self):
        """The index of the magnitude scale in the scale list"""
        return self._get_scale(self.magnitude_scales)

    @property
    def phase_index(self):
        """The index of the phase scale in the scale list"""
        return self._get_scale(self.phase_scales)

    @property
    def real_index(self):
        """The index of the real scale in the scale list"""
        return self._get_scale(self.real_scales)

    @property
    def imag_index(self):
        """The index of the imaginary scale in the scale list"""
        return self._get_scale(self.imag_scales)

    @property
    def magnitude_scales(self):
        return list(self.SUPPORTED_SCALES["magnitude"].keys())

    @property
    def phase_scales(self):
        return list(self.SUPPORTED_SCALES["phase"].keys())

    @property
    def real_scales(self):
        return list(self.SUPPORTED_SCALES["real"].keys())

    @property
    def imag_scales(self):
        return list(self.SUPPORTED_SCALES["imaginary"].keys())

    def __repr__(self):
        element_str = "%s" % self.element
        if self.scales is not None:
            element_str += ":".join(self.scales)
        return "Output[%s]" % element_str


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


class LisoNoiseSource:
    """LISO noise source"""
    def __init__(self, noise, index=None):
        if index is not None:
            index = int(index)

        self.noise = noise
        self.index = index

    def __str__(self):
        return "LISO {noise}".format(noise=self.noise)
