import sys
import os.path
import abc
import subprocess
import logging
import re
import numpy as np
from tempfile import NamedTemporaryFile

from ..data import (VoltageVoltageTF, VoltageCurrentTF, CurrentCurrentTF,
                    CurrentVoltageTF, NoiseSpectrum, Series, ComplexSeries)
from ..format import SIFormatter
from ..circuit import Circuit
from ..components import (Component, Resistor, Capacitor, Inductor, OpAmp, Input,
                          Node, CurrentNoise, VoltageNoise, JohnsonNoise)
from ..solution import Solution
from ..analysis import SmallSignalAcAnalysis

LOGGER = logging.getLogger("liso")

class InvalidLisoFileException(Exception):
    pass


class LisoOutputParser(object):
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
    
    COMMENT_REGEX = re.compile(r"^#.*?$")

    # circuit definitions
    # match text after e.g. "#2 capacitors:" and before the first line with
    # a non-whitespace character after the "#"
    COMPONENT_REGEX = re.compile(r"^#(\d+) "
                                 r"(op-amps?|capacitors?|resistors?|coils?|nodes?):"
                                 r"([\s\S]+?)(?=\n#\S+)",
                                 re.MULTILINE)

    # text to ignore in op-amp list
    OPAMP_IGNORE_STRINGS = [
        "*OVR*", # overridden parameter flag
        "s***DEFAULT", # default parameter used
        "***DEFAULT"
    ]

    # op-amp parameters
    OPAMP_REGEX = re.compile(r"^#\s+\d+\s+" # count
                             r"([\w\d]+)\s+" # name
                             r"([\w\d]+)\s+" # model
                             r"'\+'=([\w\d]+)\s+" # +in
                             r"'\-'=([\w\d]+)\s+" # -in
                             r"'out'=([\w\d]+)\s+" # out
                             r"a0=([\w\d\s\.]+)\s+" # gain
                             r"gbw=([\w\d\s\.]+)^" # gbw
                             r"\#\s+un=([\w\d\s\.]+)\/sqrt\(Hz\)\s+" # un
                             r"uc=([\w\d\s\.]+)\s+" # uc
                             r"in=([\w\d\s\.]+)\/sqrt\(Hz\)\s+" # in
                             r"ic=([\w\d\s\.]+)^" # ic
                             r"\#\s+umax=([\w\d\s\.]+)\s+" # umax
                             r"imax=([\w\d\s\.]+)\s+" # imax
                             r"sr=([\w\d\s\.]+)\/us\s+" # sr
                             r"delay=([\w\d\s\.]+)" # delay
                             r"(^\#\s+(.*)$)*", # poles/zeros (optional line)
                             re.MULTILINE)

    # op-amp roots
    OPAMP_ROOT_REGEX = re.compile(r"(pole|zero) at ([\w\d\s\.]+) "
                                  r"\((real|Q=([\w\d\s\.]+))\)")

    # data column definitions
    TF_VOLTAGE_OUTPUT_REGEX = re.compile(r"^\#OUTPUT (\d+) voltage outputs:$")
    TF_CURRENT_OUTPUT_REGEX = re.compile(r"^\#OUTPUT (\d+) current outputs:$")
    NOISE_OUTPUT_REGEX = re.compile(r"^\#Noise is computed at node ([\w\d]+) "
                                    r"for \(nnoise=(\d+), nnoisy=(\d+)\) :$")
    # "0 node: nin dB Degrees"
    TF_VOLTAGE_SINK_REGEX = re.compile(r"^\#\s*(\d+) node: ([\w\d]+) (\w+) (\w+)$")
    # "#  0 C:c2 dB Degrees"
    TF_CURRENT_SINK_REGEX = re.compile(r"^\#\s*(\d+) (\w+):([\w\d]+) (\w+) (\w+)$")
    # """#Noise is computed at node no for (nnoise=6, nnoisy=6) :
    #    #  r1 r3 r4 r6 op1(U) op1(I-) """
    NOISE_VOLTAGE_SOURCE_REGEX = re.compile(r"^\#Noise is computed at node "
                                            r"[\w\d]+ for .* :\n"
                                            r"\#\s*([\w\d\s\(\)\-\+]*)\s*$",
                                            re.MULTILINE)
    # "o1(I+)"
    NOISE_COMPONENT_REGEX = re.compile(r"^([\w\d]+)(\(([\w\d\-\+]*)\))?$")

    # input nodes
    FIXED_VOLTAGE_INPUT_NODE_REGEX = re.compile(r"\#Voltage input at node "
                                                r"(.+), impedance (.+)Ohm")
    FIXED_CURRENT_INPUT_NODE_REGEX = re.compile(r"\#Current input into node "
                                                r"(.+), impedance (.+)Ohm")
    FLOATING_VOLTAGE_INPUT_NODES_REGEX = re.compile(r"\#Floating voltage input "
                                                    r"between nodes (.+) "
                                                    r"and (.+), impedance "
                                                    r"(.+) Ohm")

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
        self._output_nodes = set()
        self._output_components = set()
        self.output_all_nodes = False
        self.output_all_opamp_nodes = False
        self.output_all_components = False
        self.output_all_opamp_components = False
        self.noise_node = None
        self.circuit = Circuit()

        # defaults
        self.data = None
        self.functions = []
        self._found_column_count = 0

        self._load_file()

    def add_output_node(self, node):
        self._output_nodes.add(node)

    def add_output_component(self, component):
        self._output_components.add(component)

    def _load_file(self):
        """Load and parse from file"""

        with open(self.filepath, "r") as obj:
            self.parse_lines(obj.readlines())

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

    def run_native(self, *args, **kwargs):
        # add input component, if not yet present
        self._add_circuit_input()

        if self.output_type == "noise":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_noise(
                frequencies=self.frequencies,
                noise_node=self.noise_node,
                *args, **kwargs)
        elif self.output_type == "tf":
            return SmallSignalAcAnalysis(circuit=self.circuit).calculate_tfs(
                frequencies=self.frequencies,
                output_components=self.output_components,
                output_nodes=self.output_nodes,
                *args, **kwargs)
        else:
            raise Exception("no outputs requested")

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

    def validate(self):
        if self.frequencies is None:
            # no frequencies found
            raise InvalidLisoFileException("no plot frequencies found (is this "
                                           "a valid LISO file?)")
        elif (self.input_node_n is None and self.input_node_p is None):
            # no input nodes found
            raise InvalidLisoFileException("no input nodes found (is this a "
                                           "valid LISO file?)")
        elif ((len(self.output_nodes) == 0 and not self.output_all_nodes)
              and (len(self.output_components) == 0 and not self.output_all_components)
              and self.noise_node is None):
            # no output requested
            raise InvalidLisoFileException("no output requested (is this a "
                                           "valid LISO file?)")

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

    @classmethod
    def tokenise(cls, line):
        """Tokenise a LISO line

        :param line: line to tokenise
        :type line: str
        :return: tokens that make up each line
        :rtype: List[str]
        """

        # strip off comments, then split into parts and remove extra whitespace
        return [line.strip() for line in line.split('#')[0].split()]

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
                           node3_name, *args):
        """Add op-amp

        :return: new op-amp
        :rtype: :class:`~OpAmp`
        """

        # add nodes first
        node1 = Node(node1_name)
        node2 = Node(node2_name)
        node3 = Node(node3_name)

        # parse extra arguments, e.g. "sr=38e6", into dict params
        extra_args = self.parse_op_amp_overrides(args)
        
        # create log message for overridden parameters
        if extra_args:
            log_extra = ", ".join(["%s=%s" % (k, v) for k, v in extra_args.items()])
        else:
            log_extra = "[none]"

        LOGGER.info("adding op-amp [%s = %s, in+ %s, in- %s, out %s, extra %s]",
                    name, model, node1, node2, node3, log_extra)

        self.circuit.add_library_opamp(name=name, model=model, node1=node1,
                                       node2=node2, node3=node3, **extra_args)

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
                input_type = "noise"

                # set input impedance
                impedance = self.input_impedance

            LOGGER.info("adding input [type %s, %s +%s -%s, R=%s]" % (input_type,
                        node, node_p, node_n, impedance))

            self.circuit.add_input(input_type=input_type, node=node,
                                   node_p=node_p, node_n=node_n,
                                   impedance=impedance)

    def set_output_type(self, output_type):
        if self.output_type is not None:
            if self.output_type != output_type:
                # output type changed
                raise Exception("output file contains both transfer functions "
                                "and noise, which is not supported")

            # output type isn't being changed; no need to do anything else
            return

        if output_type not in ["tf", "noise"]:
            raise ValueError("unknown output type")

        self.output_type = output_type
    
    @classmethod
    def parse_op_amp_overrides(cls, args):
        """Parses op-amp override strings from input file
        
        In LISO, op-amp parameters can be overridden by specifying a library parameter
        after the standard op-amp definition, e.g. "op u1 ad829 gnd n1 n2 sr=38e6"
        """

        extra_args = {}

        for arg in args:
            arg = arg.lower()

            try:
                key, value = arg.split("=")
            except ValueError:
                raise ValueError("op-amp parameter override %s invalid; must be in the "
                                 "form 'param=value'" % arg)
            
            if key not in cls.OP_OVERRIDE_MAP.keys():
                raise ValueError("unknown op-amp override parameter '%s'" % key)
            
            extra_args[cls.OP_OVERRIDE_MAP[key]] = value
        
        return extra_args

    def add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    def solution(self):
        """Get circuit solution

        :return: solution
        :rtype: :class:`~Solution`
        """

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

        # add input component before we parse columns, as current transfer
        # functions need the input component as a source
        self._add_circuit_input()

        # parse data columns
        self._parse_columns(lines)

        # check we found anything
        self.validate()

    def _parse_components(self, lines):
        text = "".join(lines)

        # find components
        for (count, description, content) in re.findall(self.COMPONENT_REGEX, text):
            if description.startswith(("resistor", "capacitor", "coil")):
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
            elif description.startswith("coil"):
                self._add_inductor(name, value, node1_name, node2_name)
            else:
                raise Exception("unrecognised component: %s" % description)

            found += 1

        if count != found:
            raise Exception("expected %d component(s), parsed %d" % (count, found))

    def _parse_opamp(self, count, description, content):
        # remove ignored strings
        content = self._remove_ignored_opamp_strings(content)

        # extract op-amp data
        matches = re.findall(self.OPAMP_REGEX, content)

        count = int(count)
        found = 0

        for (name, model, node1_name, node2_name, node3_name, a0, gbw,
             v_noise, v_corner, i_noise, i_corner, v_max, i_max, slew_rate,
             delay, _, roots) in matches:
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

    @classmethod
    def _remove_ignored_opamp_strings(cls, line):
        for ignore in cls.OPAMP_IGNORE_STRINGS:
            line = line.replace(ignore, "")
        
        return line

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
                theta = np.arccos(1 / (2 * q_factor))

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
                self.input_type = "current"
                self.input_node_p = Node(match_fixed_current.group(1))
                self.input_impedance, _ = SIFormatter.parse(match_fixed_current.group(2))

                LOGGER.info("adding fixed current input node %s with source "
                            "impedance %s", self.input_node_p,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return
            elif match_fixed_voltage:
                # fixed voltage input
                self.input_type = "voltage"
                self.input_node_p = Node(match_fixed_voltage.group(1))
                self.input_impedance, _ = SIFormatter.parse(match_fixed_voltage.group(2))

                LOGGER.info("adding fixed voltage input node %s with source "
                            "impedance %s", self.input_node_p,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return
            elif match_floating_voltage:
                # floating input
                self.input_type = "voltage"
                self.input_node_p = Node(match_floating_voltage.group(1))
                self.input_node_m = Node(match_floating_voltage.group(2))
                self.input_impedance, _ = SIFormatter.parse(match_floating_voltage.group(3))

                LOGGER.info("adding floating voltage input nodes +%s, -%s with "
                            "source impedance %s", self.input_node_p,
                            self.input_node_m,
                            SIFormatter.format(self.input_impedance, "Ω"))
                return

    def _parse_columns(self, lines):
        # reset column count
        # A global count is required for situations where both voltage and
        # current outputs are requested. In this case, the current output
        # column indices are counted from 0 even though they appear after
        # the voltage output columns.
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
        self.set_output_type("tf")

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
                                   phase=phase_data, magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            # create appropriate transfer function depending on input type
            if self.input_type == "voltage":
                function = VoltageVoltageTF(series=series, source=self.input_node_p,
                                            sink=sink)
            elif self.input_type == "current":
                function = CurrentVoltageTF(series=series,
                                            source=self.circuit.get_component("input"),
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
        self.set_output_type("tf")

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
            if self.input_type == "voltage":
                function = VoltageCurrentTF(series=series,
                                            source=self.input_node_p,
                                            sink=sink)
            elif self.input_type == "current":
                function = CurrentCurrentTF(series=series,
                                            source=self.circuit.get_component("input"),
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
        self.set_output_type("noise")

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