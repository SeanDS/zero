import os.path
import subprocess
import logging
import re
import numpy as np
from tempfile import NamedTemporaryFile

from .circuit import Circuit
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp, Node,
                         Gnd)
from .data import TransferFunction, NoiseSpectrum, Series, ComplexSeries

LOGGER = logging.getLogger("liso")

class CircuitParser(object):
    COMPONENTS = ["r", "c", "l", "op"]
    DIRECTIVES = {"input_nodes": ["uinput", "vinput"],
                  "output_nodes": ["uoutput", "voutput"],
                  "noise_node": ["noise"]}

    OUTPUT_MODE_TF = 1
    OUTPUT_MODE_NOISE = 2

    COMMENT_REGEX = re.compile("#.*?$")

    def __init__(self):
        # default values
        self.components = []
        self.nodes = {}
        self.input_nodes = []
        self.input_impedance = 0
        self.output_nodes = []
        self.noise_node = None
        self.loaded = False
        self._output_mode = None

    def load(self, filepath):
        """Load and parses a LISO .fil circuit definition file

        :param filepath: path to LISO file
        :type filepath: str
        """

        # open file
        with open(filepath, "r") as obj:
            for tokens in self._fil_tokens(obj.readlines()):
                self._parse_tokens(tokens)

        self.loaded = True

    def circuit(self):
        """Get circuit representing LISO model

        :return: circuit object
        :rtype: :class:`~Circuit`
        """

        if not self.loaded:
            raise Exception("file not loaded")

        # create empty circuit
        circuit = Circuit()

        # set defaults
        circuit.defaults["input_nodes"] = self.input_nodes
        circuit.defaults["input_impedance"] = self.input_impedance
        circuit.defaults["output_nodes"] = self.output_nodes
        circuit.defaults["noise_node"] = self.noise_node

        # add components
        for component in self.components:
            circuit.add_component(component)

        return circuit

    @property
    def output_mode(self):
        return self._output_mode

    @output_mode.setter
    def output_mode(self, mode):
        if self._output_mode is not None:
            raise ValueError("output mode already set")

        if mode is not None:
            mode = int(mode)

        self._output_mode = mode

    @classmethod
    def _fil_tokens(cls, lines):
        """Extract LISO file tokens

        :param lines: lines to parse
        :type lines: Sequence[str]
        :return: sequence of tokens that make up each line
        :rtype: Generator[List[str]]
        """

        for line in lines:
            # remove comments and extra whitespace and split into parts
            yield re.sub(cls.COMMENT_REGEX, "", line).split()

    @property
    def directives(self):
        """Get sequence of supported directives

        :return: directives
        :rtype: Generator[str]
        """

        for directive_list in self.DIRECTIVES.values():
            yield from directive_list

    def _parse_tokens(self, tokens):
        """Parse LISO tokens as commands

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
            obj = self._create_resistor(options)
        elif command == "c":
            obj = self._create_capacitor(options)
        elif command == "l":
            obj = self._create_inductor(options)
        elif command == "op":
            obj = self._create_opamp(options)
        else:
            raise ValueError("Unknown component: %s" % command)

        self.components.append(obj)

    def _create_lcr(self, _class, options):
        """Create new component

        :param _class: component class to create
        :type _class: type
        :param options: component options
        :type options: Sequence[str]
        :return: new component
        :rtype: :class:`~Component`
        """

        name = options[0]
        value = options[1]
        node1 = self._get_node(options[2])
        node2 = self._get_node(options[3])

        return _class(name=name, value=value, node1=node1, node2=node2)

    def _create_resistor(self, *options):
        """Create resistor

        :return: new resistor
        :rtype: :class:`~Resistor`
        """

        return self._create_lcr(Resistor, *options)

    def _create_capacitor(self, *options):
        """Create capacitor

        :return: new capacitor
        :rtype: :class:`~Capacitor`
        """

        return self._create_lcr(Capacitor, *options)

    def _create_inductor(self, *options):
        """Create inductor

        :return: new inductor
        :rtype: :class:`~Inductor`
        """

        return self._create_lcr(Inductor, *options)

    def _create_opamp(self, options):
        """Create op-amp

        :return: new op-amp
        :rtype: :class:`~OpAmp`
        """

        name = options[0]
        model = options[1]
        node1 = self._get_node(options[2])
        node2 = self._get_node(options[3])
        node3 = self._get_node(options[4])

        return OpAmp(name=name, model=model, node1=node1, node2=node2,
                     node3=node3)

    def _get_node(self, node_name):
        """Get node given its name

        :param node_name: name of the node
        :type node_name: str
        :return: node
        :rtype: :class:`~Node`
        """

        node_name = str(node_name)

        if node_name not in self.nodes:
            if node_name.lower() == "gnd":
                # return ground node instance
                node = Gnd()
            else:
                node = Node(node_name)

            self.nodes[node_name] = node

        return self.nodes[node_name]

    def _parse_directive(self, directive, options):
        """Parse LISO tokens as directive

        :param directive: directive string, e.g. "vinput"
        :type directive: str
        :param options: directive options
        :type options: Sequence[str]
        :raises ValueError: if directive is unknown
        """

        if directive in self.DIRECTIVES["input_nodes"]:
            self._parse_input_nodes(options)
        elif directive in self.DIRECTIVES["output_nodes"]:
            self._parse_output_nodes(options[0])
        elif directive in self.DIRECTIVES["noise_node"]:
            self._parse_noise_node(options[0])
        else:
            raise ValueError("Unknown directive: %s" % directive)

    def _parse_input_nodes(self, node_options):
        """Parse LISO token as input node directive

        :param node_options: input node options
        :type node_options: Sequence[str]
        """

        self.input_nodes = [self._get_node(node_options[0])]

        if len(node_options) > 3:
            # floating input
            self.input_nodes.append(self._get_node(node_options[1]))
            self.input_impedance = float(node_options[2])
        else:
            self.input_impedance = float(node_options[1])

    def _parse_output_nodes(self, output_str):
        """Parse LISO token as output node directive

        :param output_str: output node name, and (unused) plot scaling \
                           separated by colons
        :type output_str: str
        """

        # split options by colon
        options = output_str.split(":")

        # only use first option, which is the node name
        self.add_output_node(self._get_node(options[0]))
        # FIXME: parse list of output nodes

        # set output mode
        self.output_mode = self.OUTPUT_MODE_TF

    def _parse_noise_node(self, node_str):
        """Parse LISO token as noise node directive

        :param node_str: noise node name, and (unused) plot scaling \
                         separated by colons
        :type node_str: str
        """

        # split options by colon
        options = node_str.split(":")

        # only use first option, which is the node name
        self.noise_node = self._get_node(options[0])

        if len(options) > 1:
            LOGGER.warning("ignoring plot options in noise command")

        # set output mode
        self.output_mode = self.OUTPUT_MODE_NOISE

    def add_output_node(self, node):
        """Add output node

        :param node: output node to add
        :type node: :class:`~Node`
        """

        if node in self.output_nodes:
            raise ValueError("output node already added")

        self.output_nodes.append(node)

class Runner(object):
    """LISO runner"""

    def __init__(self, script_path):
        self.script_path = script_path

    def run(self):
        with NamedTemporaryFile() as out_path, NamedTemporaryFile() as gnu_path:
            return self._liso_result(self.script_path, out_path.name,
                                     gnu_path.name)

    def _liso_result(self, script_path, out_path, gnu_path):
        """Get LISO results

        :param script_path: path to LISO ".fil" file
        :type script_path: str
        :param out_path: path to LISO ".out" file to be created
        :type out_path: str
        :param gnu_path: path to LISO ".gnu" file to be created
        :type gnu_path: str
        :return: LISO output
        :rtype: :class:`~OutputParser`
        """

        result = self._run_liso_process(script_path, out_path, gnu_path)

        if result.returncode != 0:
            raise Exception("error during LISO run")

        return OutputParser(script_path, out_path, gnu_path)

    @staticmethod
    def _run_liso_process(script_path, out_path, gnu_path):
        input_path = os.path.abspath(script_path)

        if not os.path.exists(input_path):
            raise Exception("input file %s does not exist" % input_path)

        # LISO binary directory
        try:
            liso_dir = os.environ["LISO_DIR"]
        except KeyError:
            raise Exception("environment variable \"LISO_DIR\" must point to the "
                            "directory containing the \"fil_static\" LISO binary")

        # LISO binary path
        liso_path = os.path.join(liso_dir, "fil_static")

        # LISO flags
        flags = [input_path, out_path, gnu_path]

        # run LISO
        return subprocess.run([liso_path, *flags])

class OutputParser(object):
    """LISO output parser"""

    TF_REGEX = re.compile("using\s\(\$(\d+)\):\(\$(\d+)\).+title\s\"(.+)\((\w+)\)\s\[.+\]\s(.*)\s\"")
    NOISE_REGEX = re.compile("using\s\(\$(\d+)\):\(\$(\d+)\).+title\s\"(.+)\((\w+)\)\"")

    def __init__(self, infile, outfile, gnufile):
        # set file paths
        self.infile = infile
        self.outfile = outfile
        self.gnufile = gnufile

        # defaults
        self.functions = []

        self._parse()

    def add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    def _parse(self):
        """Parse LISO output and gnuplot files"""

        # parse circuit
        parser = CircuitParser()
        parser.load(self.infile)
        self.circuit = parser.circuit()

        # parse data file
        self.data = self._parse_outfile(self.outfile)

        # find series information using gnufile
        with open(self.gnufile, "r") as obj:
            lines = obj.readlines()

        tfs = self._match_tf_pairs(lines)
        noise = self._match_noise_spectra(lines)

        self._process_tf_pairs(tfs)
        self._process_noise_spectra(noise)

    def _process_tf_pairs(self, pairs):
        # create series from matches
        for pair in pairs.values():
            signal_type = pair["db"]["signal_type"]
            magnitude_match = pair["db"]
            phase_match = pair["phase"]

            # source is the input
            source = self.circuit.default_input_nodes[0]

            # get node or component depending on source
            if signal_type == "U":
                part = self.circuit.get_node(magnitude_match["part_name"])
            elif signal_type == "I":
                part = self.circuit.get_component(magnitude_match["part_name"])
            else:
                raise ValueError("unsupported signal type: %s" % signal_type)

            frequencies = self.data[:, magnitude_match["x_index"]]
            magnitude_data = self.data[:, magnitude_match["y_index"]]
            phase_data = self.data[:, phase_match["y_index"]]
            magnitude_scale = Series.SCALE_DB
            phase_scale = Series.SCALE_DEG

            # create data series
            series = ComplexSeries(x=frequencies, magnitude=magnitude_data,
                                   phase=phase_data,
                                   magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            function = TransferFunction(series=series,
                                        source=source,
                                        sink=part)

            self.add_function(function)

    def _process_noise_spectra(self, noise_data):
        # create series from matches
        for source_data in noise_data:
            # source is the component
            source = self.circuit.get_component(source_data["component_name"])

            # sink is the noise node
            sink = self.circuit.default_noise_node

            # noise classification
            noise_type = source_data["noise_type"]

            # get node or component depending on source
            if noise_type == "RNoise":
                noise_type = NoiseSpectrum.NOISE_JOHNSON
            elif noise_type == "OP UNoise":
                noise_type = NoiseSpectrum.NOISE_OPAMP_VOLTAGE
            elif noise_type == "OP INoise+":
                noise_type = NoiseSpectrum.NOISE_OPAMP_CURRENT
            elif noise_type == "OP INoise-":
                noise_type = NoiseSpectrum.NOISE_OPAMP_CURRENT
            else:
                raise ValueError("unsupported noise type: %s" % noise_type)

            frequencies = self.data[:, source_data["x_index"]]
            spectrum = self.data[:, source_data["y_index"]]

            # create data series
            series = Series(x=frequencies, y=spectrum)

            function = NoiseSpectrum(series=series,
                                     source=source,
                                     sink=sink,
                                     noise_type=noise_type)

            self.add_function(function)

    @staticmethod
    def _parse_outfile(filepath):
        return np.genfromtxt(filepath)

    def _match_tf_pairs(self, lines):
        """Matches gnuplot transfer functions in the specified lines

        :param lines: gnuplot lines
        :type lines: Sequence[str]
        :return: series information, grouped by pairs of magnitudes and phases
        :rtype: dict
        """

        pairs = {}

        for line in lines:
            # find noise output
            matches = re.findall(self.TF_REGEX, line)

            if len(matches) == 0:
                continue
            elif len(matches) > 1:
                raise Exception("invalid gnuplot match")

            match = matches[0]

            if len(match) != 5:
                raise Exception("invalid gnuplot match")

            x_index = int(match[0]) - 1
            y_index = int(match[1]) - 1
            signal_type = match[2]
            part_name = match[3]
            scale = match[4].lower()

            # unique identifier
            identifier = "%s(%s)" % (signal_type, part_name)

            if identifier not in pairs.keys():
                # create dict
                pairs[identifier] = {}

            data = {"x_index": x_index,
                    "y_index": y_index,
                    "signal_type": signal_type,
                    "part_name": part_name,
                    "scale": scale}

            # add this transfer function to pair
            pairs[identifier][scale] = data

        return pairs

    def _match_noise_spectra(self, lines):
        """Matches gnuplot noise spectra

        :param lines: gnuplot lines
        :type lines: Sequence[str]
        :return: series information
        :rtype: dict
        """

        noise = []

        for line in lines:
            # find noise output
            matches = re.findall(self.NOISE_REGEX, line)

            if len(matches) == 0:
                continue
            elif len(matches) > 1:
                raise Exception("invalid gnuplot match")

            match = matches[0]

            if len(match) != 4:
                raise Exception("invalid gnuplot match")

            noise.append({"x_index": int(match[0]) - 1,
                          "y_index": int(match[1]) - 1,
                          "noise_type": match[2],
                          "component_name": match[3]})

        return noise

    @property
    def frequencies(self):
        if not len(self.functions):
            raise Exception("no frequencies available (has LISO been run yet?)")

        # use first function's x-axis
        return self.functions[0].series.x
