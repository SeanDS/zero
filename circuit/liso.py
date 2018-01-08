"""LISO file parsing and running

This module provides classes to parse LISO input and output files and to run
native LISO binaries automatically. The input and output parsers implement
methods to search and identify components and commands in their respective
files.
"""

import sys
import os.path
import abc
import subprocess
import logging
import re
import numpy as np
from tempfile import NamedTemporaryFile

from .data import (VoltageVoltageTF, VoltageCurrentTF, CurrentCurrentTF,
                   CurrentVoltageTF, NoiseSpectrum, Series, ComplexSeries)
from .format import SIFormatter
from .circuit import Circuit
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp, Input,
                         Node, CurrentNoise, VoltageNoise, JohnsonNoise)
from .solution import Solution

LOGGER = logging.getLogger("liso")

class BaseParser(object, metaclass=abc.ABCMeta):
    COMMENT_REGEX = re.compile("^#.*?$")

    # output types
    TYPE_TF = 1
    TYPE_NOISE = 2

    def __init__(self, filepath):
        """Instantiate a LISO parser

        :param filepath: path to LISO file
        :type filepath: str
        """

        # file to parse
        self.filepath = filepath

        # default circuit values
        self.frequencies = None
        self.input_type = None
        self.input_node_n = None
        self.input_node_p = None
        self.input_impedance = None
        self.output_type = None
        self.output_nodes = set()
        self.output_components = set()
        self.noise_node = None
        self.circuit = Circuit()

        self._load_file()

    def add_output_node(self, node):
        self.output_nodes.add(node)

    def add_output_component(self, component):
        self.output_components.add(component)

    def _load_file(self):
        """Load and parse from file"""

        with open(self.filepath, "r") as obj:
            self.parse_lines(obj.readlines())

    @abc.abstractmethod
    def parse_lines(self, lines):
        return NotImplemented

    @abc.abstractmethod
    def solution(self):
        """Get solution"""
        return NotImplemented

    def show(self, *args, **kwargs):
        """Show LISO results"""

        if not self.plottable:
            LOGGER.warning("nothing to show")

        # get solution
        solution = self.solution(*args, **kwargs)
        # draw plots
        solution.plot(output_nodes=list(self.output_nodes),
                      output_components=list(self.output_components))
        # display plots
        solution.show()

    def run_native(self, *args, **kwargs):
        # add input component, if not yet present
        self._add_circuit_input()

        if self.output_type is self.TYPE_NOISE:
            return self.circuit.calculate_noise(
                frequencies=self.frequencies,
                noise_node=self.noise_node,
                *args, **kwargs)
        elif self.output_type is self.TYPE_TF:
            return self.circuit.calculate_tfs(
                frequencies=self.frequencies,
                output_components=self.output_components,
                output_nodes=self.output_nodes,
                *args, **kwargs)
        else:
            raise Exception("no outputs requested")

    def validate(self):
        if (self.frequencies is None or (self.input_node_n is None and
                                         self.input_node_p is None)
            or (len(self.output_nodes) == 0 and len(self.output_components) == 0
                and self.noise_node is None)):
            raise Exception("this doesn't appear to be a valid LISO file")

    @property
    def calc_tfs(self):
        return self.calc_node_tfs or self.calc_component_tfs

    @property
    def calc_node_tfs(self):
        return len(self.output_nodes) > 0

    @property
    def calc_component_tfs(self):
        return len(self.output_components) > 0

    @property
    def calc_noise(self):
        return self.noise_node is not None

    @property
    def plottable(self):
        return self.calc_tfs or self.calc_noise

    @classmethod
    def tokenise(cls, line):
        """Tokenise a LISO line

        :param line: line to tokenise
        :type line: str
        :return: tokens that make up each line
        :rtype: List[str]
        """

        # split into parts and remove extra whitespace
        return [line.strip() for line in line.split()]

    def _add_lcr(self, _class, name, value, node1_name, node2_name):
        """Add new L, C or R component

        :param _class: component class to create
        :type _class: type
        :param name: component name
        :type name: str
        :param value: component value
        :type value: float
        :param node1_name: node 1 name
        :type node1_name: str
        :param node2_name: node 2 name
        :type node2_name: str
        :return: new component
        :rtype: :class:`~Component`
        """

        node1 = Node(node1_name)
        node2 = Node(node2_name)

        LOGGER.info("adding %s [%s = %s, in %s, out %s]",
                    _class.__name__.lower(), name, value, node1, node2)

        self.circuit.add_component(_class(name=name, value=value, node1=node1,
                                          node2=node2))

    def _add_resistor(self, *args, **kwargs):
        """Add resistor

        :return: new resistor
        :rtype: :class:`~Resistor`
        """

        self._add_lcr(Resistor, *args, **kwargs)

    def _add_capacitor(self, *args, **kwargs):
        """Add capacitor

        :return: new capacitor
        :rtype: :class:`~Capacitor`
        """

        self._add_lcr(Capacitor, *args, **kwargs)

    def _add_inductor(self, *args, **kwargs):
        """Add inductor

        :return: new inductor
        :rtype: :class:`~Inductor`
        """

        self._add_lcr(Inductor, *args, **kwargs)

    def _add_opamp(self, name, model, node1_name, node2_name, node3_name, *args,
                   **kwargs):
        """Add op-amp

        :return: new op-amp
        :rtype: :class:`~OpAmp`
        """

        # add nodes first
        node1 = Node(node1_name)
        node2 = Node(node2_name)
        node3 = Node(node3_name)

        LOGGER.info("adding op-amp [%s = %s, in+ %s, in- %s, out %s]",
                    name, model, node1, node2, node3)

        self.circuit.add_component(OpAmp(name=name, model=model, node1=node1,
                                         node2=node2, node3=node3, *args,
                                         **kwargs))

    def _add_library_opamp(self, name, model, node1_name, node2_name,
                           node3_name, *args, **kwargs):
        """Add op-amp

        :return: new op-amp
        :rtype: :class:`~OpAmp`
        """

        # add nodes first
        node1 = Node(node1_name)
        node2 = Node(node2_name)
        node3 = Node(node3_name)

        LOGGER.info("adding op-amp [%s = %s, in+ %s, in- %s, out %s]",
                    name, model, node1, node2, node3)

        self.circuit.add_library_opamp(name=name, model=model, node1=node1,
                                       node2=node2, node3=node3, *args,
                                       **kwargs)

    def _add_circuit_input(self):
        # create input component
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
                input_type = Input.TYPE_NOISE

                # set input impedance
                impedance = self.input_impedance

            LOGGER.info("adding input [type %s, %s +%s -%s, R=%s]" % (input_type,
                        node, node_p, node_n, impedance))

            self.circuit.add_input(input_type=input_type, node=node,
                                   node_p=node_p, node_n=node_n,
                                   impedance=impedance)

    def set_output_type(self, output_type):
        LOGGER.debug("setting output type from %s to %s" % (self.output_type, output_type))
        if self.output_type is not None:
            if self.output_type != output_type:
                # output type changed
                raise Exception("output file contains both transfer functions "
                                "and noise, which is not supported")

            # output type isn't being changed; no need to do anything else
            return

        if output_type not in [self.TYPE_TF, self.TYPE_NOISE]:
            raise ValueError("unknown output type")

        self.output_type = output_type

class InputParser(BaseParser):
    COMPONENTS = ["r", "c", "l", "op"]
    DIRECTIVES = {"input": ["uinput", "iinput"],
                  "output": ["uoutput", "ioutput"],
                  "noise": ["noise"],
                  "frequencies": ["freq"]}

    def __init__(self, *args, **kwargs):
        super(InputParser, self).__init__(*args, **kwargs)

    @property
    def directives(self):
        """Get sequence of supported directives

        :return: directives
        :rtype: Generator[str]
        """

        for directive_list in self.DIRECTIVES.values():
            yield from directive_list

    def parse_lines(self, lines):
        """Parses a list of LISO input file lines

        :param lines: lines to parse
        :type lines: Sequence[str]
        """

        # open file
        with open(self.filepath, "r") as obj:
            for tokens in [self.tokenise(line) for line in lines
                           if not line.startswith("#")]:
                self._parse_tokens(tokens)

        # check we found anything
        self.validate()

    def _parse_tokens(self, tokens):
        """Parse LISO input file tokens as commands

        :param tokens: tokens that make up a LISO line
        :type tokens: Sequence[str]
        """

        # ignore empty lines
        if len(tokens) < 1:
            return

        command = tokens[0]

        if command in self.COMPONENTS:
            # this is a component
            self._parse_component(command, tokens[1:])
        elif command in self.directives:
            # this is a directive
            self._parse_directive(command, tokens[1:])

    def _parse_component(self, command, options):
        """Parse LISO tokens as component

        :param command: command string, e.g. "r" or "op"
        :type command: str
        :param options: tokens after command token
        :type options: Sequence[str]
        """

        if command == "r":
            self._add_resistor(*options)
        elif command == "c":
            self._add_capacitor(*options)
        elif command == "l":
            self._add_inductor(*options)
        elif command == "op":
            self._add_library_opamp(*options)
        else:
            raise ValueError("Unknown component: %s" % command)

    def _parse_directive(self, directive, options):
        """Parse LISO tokens as directive

        :param directive: directive string, e.g. "uinput"
        :type directive: str
        :param options: directive options
        :type options: Sequence[str]
        :raises ValueError: if directive is unknown
        """

        if directive in self.DIRECTIVES["input"]:
            self._parse_input(directive, options)
        elif directive in self.DIRECTIVES["output"]:
            self._parse_output(directive, options)
        elif directive in self.DIRECTIVES["noise"]:
            self._parse_noise(options[0])
        elif directive in self.DIRECTIVES["frequencies"]:
            self._parse_frequencies(options)
        else:
            raise ValueError("Unknown directive: %s" % directive)

    def _parse_input(self, input_type, options):
        """Parse LISO token as input directive

        :param input_type: input type
        :type input_type: str
        :param options: input options
        :type options: Sequence[str]
        """

        # we always have at least a positive node
        self.input_node_p = Node(options[0])

        if input_type == "uinput":
            self.input_type = Input.TYPE_VOLTAGE
            if len(options) > 2:
                # floating input
                self.input_node_n = Node(options[1])
                self.input_impedance = float(options[2])

                LOGGER.info("adding floating voltage input nodes +%s, -%s with "
                            "impedance %f", self.input_node_p,
                            self.input_node_n, self.input_impedance)
            else:
                self.input_impedance = float(options[1])

                LOGGER.info("adding voltage input node %s with impedance %s",
                            self.input_node_p,
                            SIFormatter.format(self.input_impedance, "Ω"))
        elif input_type == "iinput":
            self.input_type = Input.TYPE_CURRENT
            self.input_node_n = None
            self.input_impedance = float(options[1])

            LOGGER.info("adding current input node %s with impedance %s",
                        self.input_node_p,
                        SIFormatter.format(self.input_impedance, "Ω"))
        else:
            raise ValueError("unrecognised input type")

    def _parse_output(self, output_type, options):
        """Parse LISO token as output directive

        :param output_type: output type
        :type output_type: str
        :param options: input options
        :type options: Sequence[str]
        """

        # transfer function output
        self.set_output_type(self.TYPE_TF)

        # split options by colon
        options = options[0].split(":")

        if output_type == "uoutput":
            # only use first option, which is the node name
            node = Node(options[0])
            self.add_output_node(node)
            LOGGER.info("adding output node %s", node)
            # FIXME: parse list of output nodes
        elif output_type == "ioutput":
            # only use first option, which is the component name
            component = options[0]
            self.add_output_component(component)
            LOGGER.info("adding output component %s", component)
        else:
            raise ValueError("invalid output type")

    def _parse_noise(self, node_str):
        """Parse LISO token as noise node directive

        :param node_str: noise node name, and (unused) plot scaling \
                         separated by colons
        :type node_str: str
        """

        # noise output
        self.set_output_type(self.TYPE_NOISE)

        # split options by colon
        options = node_str.split(":")

        # only use first option, which is the node name
        node = Node(options[0])

        LOGGER.info("setting noise node %s", node)
        self.noise_node = node

        if len(options) > 1:
            LOGGER.warning("ignoring plot options in noise command")

    def _parse_frequencies(self, options):
        """Parse LISO input file frequency options

        :param options: frequency options
        :type options: Sequence[str]
        """

        if len(options) != 4:
            raise ValueError("syntax: freq lin|log start stop steps")

        start, _ = SIFormatter.parse(options[1])
        stop, _ = SIFormatter.parse(options[2])
        # steps + 1
        count = int(options[3]) + 1

        if options[0] == "lin":
            scaling_str = "linear"
            self.frequencies = np.linspace(start, stop, count)
        elif options[0] == "log":
            scaling_str = "logarithmic"
            self.frequencies = np.logspace(np.log10(start), np.log10(stop),
                                           count)
        else:
            raise ValueError("space function can be \"lin\" or \"log\"")

        LOGGER.info("simulating %i frequencies between %s and %s with %s "
                    "scaling", count, SIFormatter.format(start, "Hz"),
                    SIFormatter.format(stop, "Hz"), scaling_str)

    def solution(self, *args, **kwargs):
        """Get circuit solution

        Optional arguments are passed to :meth:`~Circuit.calculate_tfs` or
        :meth:`~Circuit.calculate_noise`.

        :return: solution
        :rtype: :class:`~Solution`
        """

        self._add_circuit_input()

        # solve
        if self.output_type is self.TYPE_TF:
            return self.circuit.calculate_tfs(
                frequencies=self.frequencies,
                output_components=self.output_components,
                output_nodes=self.output_nodes,
                *args, **kwargs)
        elif self.output_type is self.TYPE_NOISE:
            return self.circuit.calculate_noise(
                frequencies=self.frequencies,
                noise_node=self.noise_node,
                *args, **kwargs)
        else:
            raise ValueError("unrecognised output type")

class OutputParser(BaseParser):
    """LISO output parser"""

    # circuit definitions
    # match text after e.g. "#2 capacitors:" and before the first line with
    # a non-whitespace character after the "#"
    COMPONENT_REGEX = re.compile("^#(\d+) "
                                 "(op-amps?|capacitors?|resistors?|nodes?):"
                                 "([\s\S]+?)(?=\n#\S+)",
                                 re.MULTILINE)

    # op-amp parameters
    OPAMP_REGEX = re.compile("^#\s*\d+ " # count
                             "([\w\d]+) " # name
                             "([\w\d]+) " # model
                             "'\+'=([\w\d]+) " # +in
                             "'\-'=([\w\d]+) " # -in
                             "'out'=([\w\d]+) " # out
                             "a0=([\w\d\s\.]+) " # gain
                             "gbw=([\w\d\s\.]+)^" # gbw
                             "\#\s*un=([\w\d\s\.]+)\/sqrt\(Hz\) " # un
                             "uc=([\w\d\s\.]+) " # uc
                             "in=([\w\d\s\.]+)\/sqrt\(Hz\) " # in
                             "ic=([\w\d\s\.]+)^" # ic
                             "\#\s*umax=([\w\d\s\.]+) " # umax
                             "imax=([\w\d\s\.]+) " # imax
                             "sr=([\w\d\s\.]+)\/us " # sr
                             "delay=([\w\d\s\.]+)" # delay
                             "^\#\s*(.*)$", # poles / zeros
                             re.MULTILINE)

    # op-amp roots
    OPAMP_ROOT_REGEX = re.compile("(pole|zero) at ([\w\d\s\.]+) "
                                  "\((real|Q=([\w\d\s\.]+))\)")

    # data column definitions
    TF_VOLTAGE_OUTPUT_REGEX = re.compile("^\#OUTPUT (\d+) voltage outputs:$")
    TF_CURRENT_OUTPUT_REGEX = re.compile("^\#OUTPUT (\d+) current outputs:$")
    NOISE_OUTPUT_REGEX = re.compile("^\#Noise is computed at node ([\w\d]+) "
                                    "for \(nnoise=(\d+), nnoisy=(\d+)\) :$")
    # "0 node: nin dB Degrees"
    TF_VOLTAGE_SINK_REGEX = re.compile("^\#\s*(\d+) node: ([\w\d]+) (\w+) (\w+)$")
    # "#  0 C:c2 dB Degrees"
    TF_CURRENT_SINK_REGEX = re.compile("^\#\s*(\d+) (\w+):([\w\d]+) (\w+) (\w+)$")
    # """#Noise is computed at node no for (nnoise=6, nnoisy=6) :
    #    #  r1 r3 r4 r6 op1(U) op1(I-) """
    NOISE_VOLTAGE_SOURCE_REGEX = re.compile("^\#Noise is computed at node "
                                            "[\w\d]+ for .* :\n"
                                            "\#\s*([\w\d\s\(\)\-\+]*)\s*$",
                                            re.MULTILINE)
    # "o1(I+)"
    NOISE_COMPONENT_REGEX = re.compile("^([\w\d]+)(\(([\w\d\-\+]*)\))?$")

    # input nodes
    FIXED_VOLTAGE_INPUT_NODE_REGEX = re.compile("\#Voltage input at node "
                                                "([\w\d]+), impedance (\d+) Ohm")
    FIXED_CURRENT_INPUT_NODE_REGEX = re.compile("\#Current input into node "
                                                "([\w\d]+), impedance (\d+) Ohm")
    FLOATING_VOLTAGE_INPUT_NODES_REGEX = re.compile("\#Floating voltage input "
                                                    "between nodes ([\w\d]+) "
                                                    "and ([\w\d]+), impedance "
                                                    "(\d+) Ohm")

    def __init__(self, *args, **kwargs):
        # defaults
        self.data = None
        self.functions = []
        self._found_column_count = 0

        super(OutputParser, self).__init__(*args, **kwargs)

    def add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    def solution(self):
        """Get circuit solution

        :return: solution
        :rtype: :class:`~Solution`
        """

        self._add_circuit_input()

        # create solution
        solution = Solution(self.circuit, self.frequencies)

        # add functions
        for function in self.functions:
            solution.add_function(function)

        return solution

    def parse_lines(self, lines):
        # parse data
        data = np.genfromtxt(self.filepath)
        self.frequencies = data[:, 0]
        self.data = data[:, 1:]

        # parse circuit and column definitions
        self._parse_components(lines)
        self._parse_input_nodes(lines)
        self._parse_columns(lines)

        # check we found anything
        self.validate()

    def _parse_components(self, lines):
        text = "".join(lines)

        # find components
        for (count, description, content) in re.findall(self.COMPONENT_REGEX, text):
            if description.startswith(("resistor", "capacitor", "inductor")):
                self._parse_lcr(count, description, content)
            elif description.startswith("op-amp"):
                self._parse_opamp(count, description, content)
            elif description.startswith("node"):
                # nodes already defined by components
                continue

    def _parse_lcr(self, count, description, content):
        count = int(count)
        found = 0

        # tokenise non-empty lines, stripping out comment hash
        for tokens in [self.tokenise(line.lstrip("#"))
                       for line in content.splitlines() if line]:
            name = tokens[1]

            # parse value
            value, _ = SIFormatter.parse(tokens[2] + tokens[3])

            # nodes
            node1_name = tokens[4]
            node2_name = tokens[5]

            # create component
            if description.startswith("resistor"):
                self._add_resistor(name, value, node1_name, node2_name)
            elif description.startswith("capacitor"):
                self._add_capacitor(name, value, node1_name, node2_name)
            elif description.startswith("inductor"):
                self._add_inductor(name, value, node1_name, node2_name)
            else:
                raise Exception("unrecognised component: %s" % description)

            found += 1

        if count != found:
            raise Exception("expected %d component(s), parsed %d" % (count, found))

    def _parse_opamp(self, count, description, content):
        # extract op-amp data
        matches = re.findall(self.OPAMP_REGEX, content)

        count = int(count)
        found = 0

        for (name, model, node1_name, node2_name, node3_name, a0, gbw,
             v_noise, v_corner, i_noise, i_corner, v_max, i_max, slew_rate,
             delay, roots) in matches:
            # parse roots
            zeros, poles = self._parse_opamp_roots(roots)

            # convert slew rate from output file's V/us to V/s
            slew_rate, _ = SIFormatter.parse(slew_rate)
            slew_rate *= 1e6

            # create op-amp
            self._add_opamp(name, model, node1_name, node2_name, node3_name,
                            a0=a0, gbw=gbw, delay=delay, zeros=zeros,
                            poles=poles, v_noise=v_noise, i_noise=i_noise,
                            v_corner=v_corner, i_corner=i_corner, v_max=v_max,
                            i_max=i_max, slew_rate=slew_rate)

            found += 1

        if count != found:
            raise Exception("expected %d op-amp(s), parsed %d" % (count, found))

    def _parse_opamp_roots(self, roots):
        # empty roots
        zeros = []
        poles = []

        # match roots
        matches = re.findall(self.OPAMP_ROOT_REGEX, roots)

        for root, frequency, plane, q_factor in matches:
            roots = []

            # parse frequency
            frequency, _ = SIFormatter.parse(frequency)

            if plane == "real":
                roots.append(frequency)
            else:
                # calculate complex frequency using q-factor
                q_factor, _ = SIFormatter.parse(q_factor)
                theta = np.arccos(1 / (2 * qfactor))

                # add negative/positive pair of poles/zeros
                roots.append(frequency * np.exp(-1j * theta))
                roots.append(frequency * np.exp(1j * theta))

            if root == "zero":
                zeros.extend(roots)
            elif root == "pole":
                poles.extend(roots)
            else:
                raise ValueError("unrecognised root type")

        # sort ascending and return as numpy vectors
        return np.array(sorted(zeros)), np.array(sorted(poles))

    def _parse_input_nodes(self, lines):
        for line in lines:
            match_fixed_current = re.match(self.FIXED_CURRENT_INPUT_NODE_REGEX, line)
            match_fixed_voltage = re.match(self.FIXED_VOLTAGE_INPUT_NODE_REGEX, line)
            match_floating_voltage = re.match(self.FLOATING_VOLTAGE_INPUT_NODES_REGEX, line)

            if match_fixed_current:
                # fixed current input
                self.input_type = Input.TYPE_CURRENT
                self.input_node_p = Node(match_fixed_current.group(1))
                self.input_impedance = float(match_fixed_current.group(2))

                LOGGER.info("adding fixed current input node %s with source "
                            "impedance %s", self.input_node_p,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return
            elif match_fixed_voltage:
                # fixed voltage input
                self.input_type = Input.TYPE_VOLTAGE
                self.input_node_p = Node(match_fixed_voltage.group(1))
                self.input_impedance = float(match_fixed_voltage.group(2))

                LOGGER.info("adding fixed voltage input node %s with source "
                            "impedance %s", self.input_node_p,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return
            elif match_floating_voltage:
                # floating input
                self.input_type = Input.TYPE_VOLTAGE
                self.input_node_p = Node(match_floating_voltage.group(1))
                self.input_node_m = Node(match_floating_voltage.group(2))
                self.input_impedance = float(match_floating_voltage.group(3))

                LOGGER.info("adding floating voltage input nodes +%s, -%s with "
                            "source impedance %s", self.input_node_p,
                            self.input_node_m,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return

    def _parse_columns(self, lines):
        # reset column count
        # A global count is required for situations where both voltage and
        # current outputs are requested. In this case, the current output
        # column indices are count from 0 even though they appear after the
        # voltage output columns.
        self._found_column_count = 0

        for line in lines:
            voltage_match = re.match(self.TF_VOLTAGE_OUTPUT_REGEX, line)

            if voltage_match:
                self._parse_voltage_nodes(voltage_match.group(1), lines)
                continue

            current_match = re.match(self.TF_CURRENT_OUTPUT_REGEX, line)

            if current_match:
                self._parse_current_components(current_match.group(1), lines)
                continue

            noise_match = re.match(self.NOISE_OUTPUT_REGEX, line)

            if noise_match:
                self._parse_noise_components(noise_match.group(1),
                                             noise_match.group(3), lines)
                continue

    def _parse_voltage_nodes(self, count, lines):
        """Matches output file voltage transfer functions

        :param count: number of voltage outputs
        :type count: int
        :param lines: output file lines
        :type lines: Sequence[str]
        """

        count = int(count)

        assert count > 0

        # set TF output type
        self.set_output_type(self.TYPE_TF)

        # transfer function source is the input
        source = self.input_node_p

        found = 0

        # find transfer functions
        for line in lines:
            match = re.match(self.TF_VOLTAGE_SINK_REGEX, line)

            if not match:
                continue

            # data column index
            column = int(match.group(1))

            # voltage sink node
            sink = Node(match.group(2))

            # data
            magnitude_data = self.data[:, column * 2]
            phase_data = self.data[:, column * 2 + 1]

            self._found_column_count += 2

            # scales
            magnitude_scale = match.group(3)
            phase_scale = match.group(4)

            # create data series
            series = ComplexSeries(x=self.frequencies, magnitude=magnitude_data,
                                   phase=phase_data,
                                   magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            # create appropriate transfer function depending on input type
            if self.input_type is Input.TYPE_VOLTAGE:
                function = VoltageVoltageTF(series=series, source=source,
                                            sink=sink)
            elif self.input_type is Input.TYPE_CURRENT:
                function = CurrentVoltageTF(series=series, source=source,
                                            sink=sink)
            else:
                raise ValueError("unrecognised input type")
            self.add_function(function)

            # add output node
            self.add_output_node(sink)

            found += 1

        if count != found:
            raise Exception("expected %d voltage output(s), parsed %d" %
                            (count, found))

    def _parse_current_components(self, count, lines):
        """Matches output file current transfer functions

        :param count: number of current outputs
        :type count: int
        :param lines: output file lines
        :type lines: Sequence[str]
        """

        count = int(count)

        assert count > 0

        # set TF output type
        self.set_output_type(self.TYPE_TF)

        # transfer function source is the input
        source = self.input_node_p

        found = 0

        # find transfer functions
        for line in lines:
            match = re.match(self.TF_CURRENT_SINK_REGEX, line)

            if not match:
                continue

            # data column index
            column = int(match.group(1))

            # current sink component
            sink = self.circuit.get_component(match.group(3))

            # data
            magnitude_data = self.data[:, self._found_column_count + column * 2]
            phase_data = self.data[:, self._found_column_count + column * 2 + 1]

            # scales
            magnitude_scale = match.group(4)
            phase_scale = match.group(5)

            # create data series
            series = ComplexSeries(x=self.frequencies, magnitude=magnitude_data,
                                   phase=phase_data,
                                   magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            # create appropriate transfer function depending on input type
            if self.input_type is Input.TYPE_VOLTAGE:
                function = VoltageCurrentTF(series=series, source=source,
                                            sink=sink)
            elif self.input_type is Input.TYPE_CURRENT:
                function = CurrentCurrentTF(series=series, source=source,
                                            sink=sink)
            else:
                raise ValueError("unrecognised input type")
            # add transfer function
            self.add_function(function)

            # add output node
            self.add_output_component(sink)

            found += 1

        if count != found:
            raise Exception("expected %d current output(s), parsed %d" %
                            (count, found))

    def _parse_noise_components(self, node_name, count, lines):
        """Matches output file noise spectra

        :param node_name: name of noise sink node
        :type node_name: str
        :param count: number of noise sources
        :type count: int
        :param lines: output file lines
        :type lines: Sequence[str]
        """

        count = int(count)

        assert count > 0

        # set noise output type
        self.set_output_type(self.TYPE_NOISE)

        # find noise component information
        matches = re.search(self.NOISE_VOLTAGE_SOURCE_REGEX, "".join(lines))

        # noise sink is the noise node
        self.noise_node = Node(node_name)

        # split into list
        source_strs = matches.group(1).split()

        found = 0

        for index, source_str in enumerate(source_strs):
            # extract noise source
            source = self._parse_noise_source(source_str)

            spectrum = self.data[:, index]

            # create data series
            series = Series(x=self.frequencies, y=spectrum)

            self.add_function(NoiseSpectrum(source=source,
                                            sink=self.noise_node,
                                            series=series))

            found += 1

        if count != found:
            raise Exception("expected %d noise source(s), parsed %d" %
                            (count, found))

    def _parse_noise_source(self, source_str):
        # get rid of whitespace around string
        source_str = source_str.strip()

        # look for component name and brackets
        match = re.match(self.NOISE_COMPONENT_REGEX, source_str)

        component = self.circuit.get_component(match.group(1))

        # component noise type, e.g. I+ (or empty, for resistors)
        noise_str = match.group(3)

        # group 2 is not empty if noise type is specified
        if match.group(2):
            # op-amp noise; check first character
            if noise_str[0] == "U":
                noise_source = VoltageNoise(component=component)
            elif noise_str[0] == "I":
                # work out node
                if noise_str[1] == "+":
                    # non-inverting node
                    node = component.node1
                elif noise_str[1] == "-":
                    # inverting node
                    node = component.node2
                else:
                    raise ValueError("unexpected current noise node")

                noise_source = CurrentNoise(node=node, component=component)
            else:
                raise ValueError("unrecognised noise source")
        else:
            noise_source = JohnsonNoise(component=component,
                                        resistance=component.resistance)

        return noise_source

class Runner(object):
    """LISO runner"""

    def __init__(self, script_path):
        self.script_path = script_path

    def run(self, plot=False, liso_path=None, output_path=None):
        self.liso_path = liso_path

        if not output_path:
            temp_file = NamedTemporaryFile()
            output_path = temp_file.name

        return self._liso_result(self.script_path, output_path, plot)

    def _liso_result(self, script_path, output_path, plot):
        """Get LISO results

        :param script_path: path to LISO ".fil" file
        :type script_path: str
        :param output_path: path to LISO ".out" file to be created
        :type output_path: str
        :param plot: whether to show result with gnuplot
        :type plot: bool
        :return: LISO output
        :rtype: :class:`~OutputParser`
        """

        result = self._run_liso_process(script_path, output_path, plot)

        if result.returncode != 0:
            raise Exception("error during LISO run")

        LOGGER.debug("parsing LISO output")
        return OutputParser(output_path)

    def _run_liso_process(self, script_path, output_path, plot):
        input_path = os.path.abspath(script_path)

        if not os.path.exists(input_path):
            raise Exception("input file %s does not exist" % input_path)

        # LISO flags
        flags = [input_path, output_path]

        # plotting
        if not plot:
            flags.append("-n")

        liso_path = self.liso_path
        LOGGER.debug("running LISO binary at %s", liso_path)

        # run LISO
        return subprocess.run([liso_path, *flags], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)

    @property
    def liso_path(self):
        if self._liso_path is not None:
            return self._liso_path

        # use environment variable
        try:
            liso_dir = os.environ["LISO_DIR"]
        except KeyError:
            raise Exception("environment variable \"LISO_DIR\" must point to the "
                            "directory containing the LISO binary")

        return self.find_liso(liso_dir)

    @liso_path.setter
    def liso_path(self, path):
        self._liso_path = path

    @staticmethod
    def find_liso(directory):
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            # in order of preference
            filenames = ["fil_static", "fil"]
        elif sys.platform.startswith("win32"):
            filenames = ["fil.exe"]
        else:
            raise EnvironmentError("unrecognised operating system")

        for filename in filenames:
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                return path

        raise FileNotFoundError("no appropriate LISO binary found")
