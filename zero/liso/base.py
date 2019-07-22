"""Base LISO parser"""

import sys
import os
import abc
import logging
from ply import lex, yacc
import numpy as np

from ..circuit import Circuit, ElementNotFoundError
from ..components import Node, OpAmp
from ..analysis import AcSignalAnalysis, AcNoiseAnalysis
from ..data import MultiNoiseDensity
from ..format import Quantity
from ..misc import ChangeFlagDict

LOGGER = logging.getLogger(__name__)


class LisoParserError(ValueError):
    """LISO parser error"""
    def __init__(self, message, line=None, pos=None, **kwargs):
        if line is not None:
            line = int(line)

            if pos is not None:
                pos = int(pos)

                # add line number and position
                message = f"{message} (line {line}, position {pos})"
            else:
                # add line number
                message = f"{message} (line {line})"

        # prepend message
        message = f"LISO syntax error: {message}"

        super().__init__(message, **kwargs)


class LisoParser(metaclass=abc.ABCMeta):
    """Base LISO parser"""
    def __init__(self):
        # Circuit object and the properties from which it is built.
        self.circuit = None
        self._circuit_properties = None

        # Circuit model built flag.
        self._circuit_built = False

        # Circuit solution.
        self._solution = None

        # Initial line and character positions.
        self.lineno = None
        self._previous_newline_position = None

        # Whether parser end of file has been reached.
        self._eof = None

        # Initialise parser properties.
        self.reset()

        # Create lexer and parser handlers. Set lex and yacc to not generate grammar files, for
        # packaging simplicity, at the cost of a slight speed penalty.
        self.lexer = lex.lex(module=self, optimize=False, debug=False)
        self.parser = yacc.yacc(module=self, write_tables=False, debug=False)

    def reset(self):
        """Reset parser to default state."""
        self.circuit = Circuit()

        self._circuit_properties = ChangeFlagDict(self._default_circuit_properties)

        self._circuit_built = False
        self._solution = None
        self.lineno = 1
        self._previous_newline_position = 0
        self._eof = False

    @property
    def _default_circuit_properties(self):
        """Default properties assumed before any circuit definitions are parsed."""
        return {"frequencies": None,
                "input_type": None,
                "input_node_p": None,
                "input_node_n": None,
                "input_impedance": None,
                "output_type": None,
                "input_refer": False,
                "noise_output_element": None,
                "response_outputs": [],
                # Displayed noise.
                "noisy_elements": [],
                # Extra noise to include in "sum" in addition to displayed noise.
                "noisy_sum_elements": [],
                # List of (name, coupling, l1, l2) inductor coupling tuples.
                "inductor_couplings": [],
                # Flag for when noise sum must be computed when building solution.
                "noise_sum_to_be_computed": False}

    @property
    def parsing_started(self):
        return self._circuit_properties.changed

    @property
    def frequencies(self):
        """Analysis frequencies"""
        return self._circuit_properties["frequencies"]

    @frequencies.setter
    def frequencies(self, frequencies):
        if self.frequencies is not None:
            self.p_error("cannot redefine frequencies")

        self._circuit_properties["frequencies"] = np.array(frequencies)

    @property
    def input_type(self):
        """Circuit input type"""
        return self._circuit_properties["input_type"]

    @input_type.setter
    def input_type(self, input_type):
        if self.input_type is not None:
            self.p_error("cannot redefine input type")

        self._circuit_properties["input_type"] = input_type

    @property
    def input_node_p(self):
        """Circuit positive input node"""
        return self._circuit_properties["input_node_p"]

    @input_node_p.setter
    def input_node_p(self, input_node_p):
        if self.input_node_p is not None:
            self.p_error("cannot redefine positive input node")

        self._circuit_properties["input_node_p"] = Node(input_node_p)

    @property
    def input_node_n(self):
        """Circuit negative input node"""
        return self._circuit_properties["input_node_n"]

    @input_node_n.setter
    def input_node_n(self, input_node_n):
        if self.input_node_n is not None:
            self.p_error("cannot redefine negative input node")

        self._circuit_properties["input_node_n"] = Node(input_node_n)

    @property
    def input_impedance(self):
        """Circuit input impedance"""
        return self._circuit_properties["input_impedance"]

    @input_impedance.setter
    def input_impedance(self, input_impedance):
        if self.input_impedance is not None:
            self.p_error("cannot redefine input impedance")

        self._circuit_properties["input_impedance"] = Quantity(input_impedance, "Ω")

    @property
    def noise_output_element(self):
        """Node or component that noise sources in the circuit are projected to.

        Note: this is a string.
        """
        return self._circuit_properties["noise_output_element"]

    @noise_output_element.setter
    def noise_output_element(self, noise_output_element):
        if self.noise_output_element is not None:
            self.p_error("cannot redefine noise output element")
        self._circuit_properties["noise_output_element"] = noise_output_element

    @property
    def response_outputs(self):
        return self._circuit_properties["response_outputs"]

    def add_response_output(self, output):
        """Add response output.

        This stores the specified sink for use in building the solution.

        Parameters
        ----------
        sink : :class:`LisoOutputElement`
            The sink to add.

        Raises
        ------
        ValueError
            If the specified sink is already present.
        """
        if output in self.response_outputs:
            raise ValueError(f"sink '{output}' is already present")

        self._circuit_properties["response_outputs"].append(output)

    @property
    def inductor_couplings(self):
        return self._circuit_properties["inductor_couplings"]

    @property
    def n_response_outputs(self):
        return len(self.response_outputs)

    @property
    def n_displayed_noise(self):
        return len(self.displayed_noise_objects)

    @property
    def n_summed_noise(self):
        return len(self.summed_noise_objects)

    def parse(self, text=None, path=None):
        """Parse LISO file.

        Parameters
        ----------
        text : :class:`str`, optional
            LISO text to parse.
        path : :class:`str`, optional
            Path to LISO file to parse.
        """
        if text is None and path is None:
            raise ValueError("must provide either text or a path")

        if path is not None:
            if text is not None:
                raise ValueError("cannot specify both text and a file to parse")

            if not os.path.isfile(path):
                raise FileNotFoundError(f"cannot read '{path}'")

            with open(path, "r") as obj:
                text = obj.read()

        # reset end of file
        self._eof = False

        self.parser.parse(text, lexer=self.lexer)

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        pos = t.lexer.lexpos - self._previous_newline_position
        raise LisoParserError(f"illegal character '{t.value[0]}'", self.lineno, pos)

    @abc.abstractmethod
    def p_error(self, p):
        """Child classes must implement error handler"""
        raise NotImplementedError

    def default_response_sources(self):
        """Default response sources"""
        # note: this cannot be a property, otherwise lex will execute the property before
        # input_type is set.
        if self.input_type == "voltage":
            sources = [self.input_node_p]
        elif self.input_type == "current":
            sources = ["input"]
        else:
            raise ValueError("unrecognised input type")

        return sources

    def default_response_sinks(self):
        """Default response sinks.

        Returns
        -------
        :class:`list` of :class:`Component` or :class:`Node`
        """
        return [element.element for element in self.response_outputs]

    def solution(self, force=False, set_default_plots=True, **kwargs):
        """Get the solution to the analysis defined in the parsed file.

        Parameters
        ----------
        force : :class:`bool`
            Whether to force the solution to be recomputed if already generated.
        set_default_plots : :class:`bool`
            Set the plots defined in the LISO file as defaults.
        """
        if not self.parsing_started:
            raise LisoParserError("no circuit defined")

        # build circuit if necessary
        self.build()

        if not self._solution or force:
            self._solution = self._run(**kwargs)

            if self._circuit_properties["noise_sum_to_be_computed"]:
                # Find spectra in solution.
                sum_spectra_groups = self._solution.filter_noise(sources=self.summed_noise_objects)
                sum_spectra = [spectral_density for group_spectra in sum_spectra_groups.values()
                               for spectral_density in group_spectra]
                # Get sink element.
                if self.input_refer:
                    if self.input_type == "voltage":
                        sum_sink = self.input_node_p
                    elif self.input_type == "current":
                        sum_sink = self.circuit["input"]
                    else:
                        raise ValueError("invalid input type")
                else:
                    sum_sink = self.circuit[self.noise_output_element]
                # Create overall spectral density.
                sum_spectral_density = MultiNoiseDensity(sink=sum_sink, constituents=sum_spectra)
                # Build noise sum. Sums are always shown by default.
                self._solution.add_noise_sum(sum_spectral_density, default=True)

        if set_default_plots:
            self._set_default_plots()

        return self._solution

    @property
    def noisy_elements(self):
        return self._circuit_properties["noisy_elements"]

    def add_noisy_element(self, noisy_element):
        """Add noise source.

        This stores the specified noise source for use in building the solution.

        Parameters
        ----------
        noisy_element : :class:`LisoNoiseSource`
            The noise source to add.

        Raises
        ------
        ValueError
            If the specified noise source is already present.
        """
        if noisy_element in self.noisy_elements:
            raise ValueError(f"noise source '{noisy_element}' is already present")

        self._circuit_properties["noisy_elements"].append(noisy_element)

    @property
    def noisy_sum_elements(self):
        return self._circuit_properties["noisy_sum_elements"]

    def add_noisy_sum_element(self, noisy_sum_element):
        """Add noise sum source.

        This stores the specified noise source for use in building the sum function in the solution.

        Parameters
        ----------
        noisy_sum_element : :class:`LisoNoiseSource`
            The noise source to add to sum.

        Raises
        ------
        ValueError
            If the specified noise sum source is already present.
        """
        if noisy_sum_element in self.noisy_sum_elements:
            raise ValueError(f"noise sum source '{noisy_sum_element}' is already present")

        self._circuit_properties["noisy_sum_elements"].append(noisy_sum_element)

    @property
    def displayed_noise_objects(self):
        """Noise sources to be plotted"""
        return self._get_noise_objects(self.noisy_elements)

    @property
    def summed_noise_objects(self):
        """Noise sources included in the sum column"""
        return self._get_noise_objects(self.noisy_sum_elements)

    def _get_noise_objects(self, noisy_sources):
        """Get noise objects for the specified noisy sources."""
        sources = set()

        for noisy_source in noisy_sources:
            component_name = noisy_source.component

            if component_name == "all":
                # Show all noise sources.
                for component in self.circuit.components:
                    sources.update(self._get_component_noise(component, noisy_source))
            elif component_name == "allop":
                # Show all op-amp noise sources.
                for opamp in self.circuit.opamps:
                    sources.update(self._get_component_noise(opamp, noisy_source))
            elif component_name == "allr":
                # Show all resistor noise sources.
                if noisy_source.has_suffix:
                    self.p_error("cannot specify noise type for 'allr'")

                for resistor in self.circuit.resistors:
                    sources.update(self._get_component_noise(resistor, noisy_source))
            elif component_name == "sum":
                # Show sum of circuit noises.
                self._circuit_properties["noise_sum_to_be_computed"] = True
            else:
                # This is a single component.
                component = self.circuit[component_name]
                sources.update(self._get_component_noise(component, noisy_source))

        return sources

    def _get_component_noise(self, component, noisy_source=None):
        """Get noise list from specified component given the optionally specified noisy source.

        If specified, the noisy source is used to define which noise sources are returned for
        op-amps.
        """
        if noisy_source is not None and noisy_source.has_suffix:
            if not isinstance(component, OpAmp):
                self.p_error("noise suffices cannot be specified on non-op-amps")

            noise = []

            # Get user-defined noise types.
            if noisy_source.has_opamp_voltage_noise and component.has_voltage_noise:
                noise.append(component.voltage_noise)
            if noisy_source.has_opamp_non_inv_current_noise and component.has_non_inv_current_noise:
                noise.append(component.non_inv_current_noise)
            if noisy_source.has_opamp_inv_current_noise and component.has_inv_current_noise:
                noise.append(component.inv_current_noise)
        else:
            # All noise types.
            noise = component.noise

        return noise

    def _set_default_plots(self):
        """Set functions that are displayed by default when the solution is plotted."""
        if self.output_type == "response":
            sources = self.default_response_sources()
            sinks = self.default_response_sinks()
            default_responses = self._solution.filter_responses(sources=sources, sinks=sinks)
            for group, responses in default_responses.items():
                for response in responses:
                    if not self._solution.is_default_response(response, group):
                        self._solution.set_response_as_default(response, group)
        elif self.output_type == "noise":
            default_spectra = self._solution.filter_noise(sources=self.displayed_noise_objects)
            for group, spectra in default_spectra.items():
                for spectral_density in spectra:
                    if not self._solution.is_default_noise(spectral_density, group):
                        self._solution.set_noise_as_default(spectral_density, group)

    def _get_analysis(self, **kwargs):
        if self.output_type == "response":
            return AcSignalAnalysis(circuit=self.circuit, **kwargs)
        elif self.output_type == "noise":
            return AcNoiseAnalysis(circuit=self.circuit, **kwargs)
        self.p_error("no outputs requested")

    def _run(self, print_progress=False, stream=None, **kwargs):
        # Build circuit if necessary.
        self.build()

        analysis = self._get_analysis(print_progress=print_progress, stream=stream)
        analysis_args = {'frequencies': self.frequencies}

        if self.input_node_n is None:
            # Grounded input.
            analysis_args['node'] = self.input_node_p
        else:
            # Floating input.
            analysis_args['node_n'] = self.input_node_n
            analysis_args['node_p'] = self.input_node_p

        if self.output_type == "noise":
            analysis_args['sink'] = self.circuit[self.noise_output_element]
            analysis_args['impedance'] = self.input_impedance
            analysis_args['input_refer'] = self.input_refer

        return analysis.calculate(self.input_type, **analysis_args, **kwargs)

    def build(self):
        """Build circuit if not yet built"""
        if not self._circuit_built:
            self._do_build()

            # check the circuit is valid
            self._validate()

            self._circuit_built = True

    def _do_build(self):
        """Build circuit"""
        # Unset lineno to avoid using the last line in subsequent errors.
        self.lineno = None

        # Coupling between inductors.
        self._set_inductor_couplings()

    def _validate(self):
        if self.frequencies is None:
            # no frequencies found
            self.p_error("no plot frequencies found")
        elif self.input_node_n is None and self.input_node_p is None:
            # no input nodes found
            self.p_error("no input nodes found")
        elif not self.response_outputs and self.noise_output_element is None:
            # no output requested
            self.p_error("no output requested")

        # check for invalid output nodes
        for response_output in self.response_outputs:
            # get element
            element = response_output.element

            if not self.circuit.has_component(element) and not self.circuit.has_node(element):
                self.p_error(f"output element '{element}' is not present in the circuit")

        # noise output element must exist
        if self.noise_output_element is not None:
            if self.noise_output_element not in self.circuit:
                self.p_error(f"noise output element '{self.noise_output_element}' is not present "
                             "in the circuit")

        # check if noise sources exist
        try:
            _ = self.displayed_noise_objects
            _ = self.summed_noise_objects
        except ElementNotFoundError as e:
            self.p_error(f"noise source '{e.name}' is not present in the circuit")

        # sum cannot be computed without a "noisy" command
        if self._circuit_properties["noise_sum_to_be_computed"] and not self.summed_noise_objects:
            # no sum noises defined
            self.p_error("noise sum requires noisy components to be defined")

    @property
    def will_calc_responses(self):
        """Whether the analysis will calculate responses"""
        return self.will_calc_node_responses or self.will_calc_component_responses

    @property
    def will_calc_node_responses(self):
        """Whether the analysis will calculate responses to nodes"""
        return len(self.output_nodes) > 0

    @property
    def will_calc_component_responses(self):
        """Whether the analysis will calculate responses to components"""
        return len(self.output_components) > 0

    @property
    def will_calc_noise(self):
        """Whether the analysis will calculate noise"""
        return self.noise_output_element is not None

    @property
    def plottable(self):
        """Whether the analysis will calculate something that can be plotted"""
        return self.will_calc_responses or self.will_calc_noise

    @property
    def output_nodes(self):
        """The output nodes of the responses computed in the analysis"""
        return set([element.element for element in self.response_outputs
                    if element.type == "node"])

    @property
    def output_components(self):
        """The output components of the responses computed in the analysis"""
        return set([element.element for element in self.response_outputs
                    if element.type == "component"])

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
        """Response output type"""
        return self._circuit_properties["output_type"]

    @output_type.setter
    def output_type(self, output_type):
        if self.output_type is not None:
            if self.output_type == output_type:
                # output type isn't being changed; no need to do anything else
                return

            # output type changed
            self.p_error("output file contains both responses and noise, which is not supported")

        if output_type not in ["response", "noise"]:
            raise ValueError("unknown output type")

        self._circuit_properties["output_type"] = output_type

    @property
    def input_refer(self):
        """Whether noise is project to the input."""
        return self._circuit_properties["input_refer"]

    @input_refer.setter
    def input_refer(self, input_refer):
        self._circuit_properties["input_refer"] = bool(input_refer)

    def _set_inductor_couplings(self):
        # discard name (not used in circuit mode)
        for _, value, inductor_1, inductor_2 in self.inductor_couplings:
            self.circuit.set_inductor_coupling(inductor_1, inductor_2, value)


class LisoOutputElement(metaclass=abc.ABCMeta):
    """LISO output element"""
    # supported scales
    SUPPORTED_SCALES = {"magnitude": {"db": ["db"], "abs": ["abs"]},
                        "phase": {"deg": ["deg", "degrees", "degrees (>0)", "degrees (<0)",
                                          "degrees (continuous)"]},
                        "real": {"real": ["re", "real"]},
                        "imaginary": {"imag": ["im", "imag"]}}

    OUTPUT_TYPE = None

    def __init__(self, type_, element=None, scales=None, index=None):
        if scales is None:
            scales = []

        if index is not None:
            index = int(index)

        self._scales = None

        self.type = str(type_)
        self.element = element
        self.scales = scales
        self.index = index

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

        raise ValueError(f"unrecognised scale: '{raw_scale}'")

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
        element_str = f"{self.element}"
        if self.scales is not None:
            element_str += ":".join(self.scales)
        return f"Output[{element_str}]"


class LisoOutputVoltage(LisoOutputElement):
    """LISO output voltage"""
    OUTPUT_TYPE = "voltage"

    def __init__(self, node=None, **kwargs):
        super().__init__("node", element=node, **kwargs)

    @property
    def node(self):
        """The output voltage node"""
        return self.element


class LisoOutputCurrent(LisoOutputElement):
    """LISO output current"""
    OUTPUT_TYPE = "current"

    def __init__(self, component=None, **kwargs):
        super().__init__("component", element=component, **kwargs)

    @property
    def component(self):
        """The output current component"""
        return self.element


class LisoNoisyElement:
    """LISO noisy element"""
    # Op-amp noise types.
    OPAMP_NOISE_TYPE_VOLTAGE = 0
    OPAMP_NOISE_TYPE_NON_INV_CURRENT = 1
    OPAMP_NOISE_TYPE_INV_CURRENT = 2

    def __init__(self, component, suffix=None, index=None):
        if index is not None:
            index = int(index)

        self._suffices = None
        self.component = component
        self.index = index

        self._parse_suffices(suffix)

    def _parse_suffices(self, suffix):
        if suffix is None:
            return

        try:
            single_suffix = int(suffix)
        except ValueError:
            str_suffix = str(suffix).lower()
            if str_suffix in ["u", "i+", "i-"]:
                single_suffix = str_suffix
            else:
                # This is not an output suffix.
                single_suffix = None

        if single_suffix is not None:
            if single_suffix in [0, "u"]:
                self._suffices = [self.OPAMP_NOISE_TYPE_VOLTAGE]
            elif single_suffix in [1, "i+"]:
                self._suffices = [self.OPAMP_NOISE_TYPE_NON_INV_CURRENT]
            elif single_suffix in [2, "i-"]:
                self._suffices = [self.OPAMP_NOISE_TYPE_INV_CURRENT]
            else:
                raise ValueError(f"unrecognised noise suffix '{single_suffix}'")
            return

        # Potentially multiple suffices specified.
        suffices = []

        for char in suffix:
            char = char.lower()
            if char == "u":
                suffices.append(self.OPAMP_NOISE_TYPE_VOLTAGE)
            elif char == "+":
                suffices.append(self.OPAMP_NOISE_TYPE_NON_INV_CURRENT)
            elif char == "-":
                suffices.append(self.OPAMP_NOISE_TYPE_INV_CURRENT)
            else:
                raise ValueError(f"unrecognised noise suffix '{char}'")

        self._suffices = suffices

    @property
    def has_suffix(self):
        """Whether a noise type suffix has been defined for this noisy element."""
        return self._suffices is not None

    @property
    def has_opamp_voltage_noise(self):
        return self.has_suffix and self.OPAMP_NOISE_TYPE_VOLTAGE in self._suffices

    @property
    def has_opamp_non_inv_current_noise(self):
        return self.has_suffix and self.OPAMP_NOISE_TYPE_NON_INV_CURRENT in self._suffices

    @property
    def has_opamp_inv_current_noise(self):
        return self.has_suffix and self.OPAMP_NOISE_TYPE_INV_CURRENT in self._suffices

    def __str__(self):
        if self.has_suffix:
            suffix_list = ", ".join(self._suffices)
            suffix_str = f", [{suffix_list}]"
        else:
            suffix_str = ""
        return f"Noise[{self.component}{suffix_str}]"
