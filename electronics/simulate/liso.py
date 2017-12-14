import os.path
import subprocess
import logging
import re
import numpy as np
from tempfile import NamedTemporaryFile

from ..data import (VoltageTransferFunction, CurrentTransferFunction,
                    NoiseSpectrum, Series, ComplexSeries)
from ..format import SIFormatter
from .circuit import Circuit
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp, Node,
                         Gnd)

LOGGER = logging.getLogger("liso")

class CircuitParser(object):
    COMPONENTS = ["r", "c", "l", "op"]
    DIRECTIVES = {"input_nodes": ["uinput", "vinput"],
                  "output_nodes": ["uoutput", "voutput"],
                  "noise_node": ["noise"],
                  "frequencies": ["freq"]}

    OUTPUT_MODE_TF = 1
    OUTPUT_MODE_NOISE = 2

    COMMENT_REGEX = re.compile("#.*?$")

    def __init__(self):
        # default values
        self.frequencies = None
        self.components = []
        self.nodes = {}
        self.input_nodes = []
        self.input_impedance = 0
        self.output_nodes = []
        self.noise_node = None
        self.loaded = False
        self._output_mode = None
        self.solution = None

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

    def run(self, *args, **kwargs):
        """Run LISO script

        Optional arguments are passed to :meth:`~Circuit.solve`.
        """

        # solve
        self.solution = self.circuit().solve(frequencies=self.frequencies,
                                             input_nodes=self.input_nodes,
                                             input_impedance=self.input_impedance,
                                             noise_node=self.noise_node,
                                             *args, **kwargs)

    def show(self):
        """Show LISO results"""

        if self.calc_tfs:
            self.solution.plot_tf(output_nodes=self.output_nodes)

        if self.calc_noise:
            self.solution.plot_noise()

        # display plots
        self.solution.show()

    def circuit(self):
        """Get circuit representing LISO model

        :return: circuit object
        :rtype: :class:`~Circuit`
        """

        if not self.loaded:
            raise Exception("file not loaded")

        # create empty circuit
        circuit = Circuit()

        # add components
        for component in self.components:
            circuit.add_component(component)

        return circuit

    def get_component(self, component_name):
        """Get circuit component by name

        :param component_name: name of component to fetch
        :type component_name: str
        :return: component
        :rtype: :class:`~Component`
        :raises ValueError: if component not found
        """

        for component in self.components:
            if component.name == component_name:
                return component

        raise ValueError("component not found")

    def get_node(self, node_name):
        """Get circuit node by name

        :param node_name: name of node to fetch
        :type node_name: str
        :return: node
        :rtype: :class:`~Node`
        :raises ValueError: if node not found
        """

        for component in self.components:
            for node in component.nodes:
                if node.name == node_name:
                    return node

        raise ValueError("node not found")

    @property
    def calc_tfs(self):
        return len(self.output_nodes) > 0

    @property
    def calc_noise(self):
        return self.noise_node is not None

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
        elif directive in self.DIRECTIVES["frequencies"]:
            self._parse_frequencies(options)
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

    def _parse_frequencies(self, options):
        """Parse LIGO token as frequency settings

        :param options: frequency options
        :type options: Sequence[str]
        """

        if len(options) != 4:
            raise ValueError("syntax: freq lin|log start stop steps")

        start = SIFormatter.parse(options[1])
        stop = SIFormatter.parse(options[2])
        # steps + 1
        count = int(options[3]) + 1

        if options[0] == "lin":
            self.frequencies = np.linspace(start, stop, count)
        elif options[0] == "log":
            self.frequencies = np.logspace(np.log10(start), np.log10(stop),
                                           count)
        else:
            raise ValueError("space function can be \"lin\" or \"log\"")

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

    def run(self, plot=False):
        with NamedTemporaryFile() as out_path:
            return self._liso_result(self.script_path, out_path.name, plot)

    def _liso_result(self, script_path, out_path, plot):
        """Get LISO results

        :param script_path: path to LISO ".fil" file
        :type script_path: str
        :param out_path: path to LISO ".out" file to be created
        :type out_path: str
        :param plot: whether to show result with gnuplot
        :type plot: bool
        :return: LISO output
        :rtype: :class:`~OutputParser`
        """

        result = self._run_liso_process(script_path, out_path, plot)

        if result.returncode != 0:
            raise Exception("error during LISO run")

        return OutputParser(script_path, out_path)

    @staticmethod
    def _run_liso_process(script_path, out_path, plot):
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
        flags = [input_path, out_path]

        # plotting
        if not plot:
            flags.append("-n")

        # run LISO
        return subprocess.run([liso_path, *flags])

class OutputParser(object):
    """LISO output parser"""

    TF_VOLTAGE_OUTPUT_REGEX = re.compile("^\#OUTPUT (\d+) voltage outputs:$")
    TF_CURRENT_OUTPUT_REGEX = re.compile("^\#OUTPUT (\d+) current outputs:$")
    NOISE_OUTPUT_REGEX = re.compile("^\#Noise is computed at node ([\w\d]+) for .* :$")
    # "0 node: nin dB Degrees"
    TF_VOLTAGE_SINK_REGEX = re.compile("^\#\s*(\d+) node: ([\w\d]+) (\w+) (\w+)$")
    # "#  0 C:c2 dB Degrees"
    TF_CURRENT_SINK_REGEX = re.compile("^\#\s*(\d+) (\w+):([\w\d]+) (\w+) (\w+)$")
    # """#Noise is computed at node no for (nnoise=6, nnoisy=6) :
    #    #  r1 r3 r4 r6 op1(U) op1(I-) """
    NOISE_VOLTAGE_SOURCE_REGEX = re.compile("^\#Noise is computed at node [\w\d]+ for .* :\n\#\s*([\w\d\s\(\)\-\+]*)\s*$",
                                            re.MULTILINE)
    # "o1(I+)"
    NOISE_COMPONENT_REGEX = re.compile("^([\w\d]+)(\(([\w\d\-\+]*)\))?$")

    def __init__(self, infile, outfile):
        # set file paths
        self.infile = infile
        self.outfile = outfile

        # defaults
        self.parser = None
        self.functions = []
        self.data = None

        self._parse()

    def add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    def _parse(self):
        """Parse LISO input and output files"""

        # parse circuit
        self.parser = CircuitParser()
        self.parser.load(self.infile)

        # parse output data
        self._parse_outfile_data(self.outfile)

    def _parse_outfile_data(self, filepath):
        # parse data
        self.data = np.genfromtxt(filepath)

        with open(filepath, "r") as obj:
            lines = obj.readlines()

        for line in lines:
            if re.match(self.TF_VOLTAGE_OUTPUT_REGEX, line):
                self._parse_voltage_nodes(lines)
            elif re.match(self.TF_CURRENT_OUTPUT_REGEX, line):
                self._parse_current_components(lines)
            elif re.match(self.NOISE_OUTPUT_REGEX, line):
                self._parse_noise_components(lines)

    def _parse_voltage_nodes(self, lines):
        """Matches output file voltage transfer functions

        :param lines: output file lines
        :type lines: Sequence[str]
        """

        # transfer function source is the input
        source = self.parser.input_nodes[0]

        # find transfer functions
        for line in lines:
            match = re.match(self.TF_VOLTAGE_SINK_REGEX, line)

            if not match:
                continue

            # data column index
            column = int(match.group(1))

            # voltage sink node
            sink = self.parser.get_node(match.group(2))

            # data
            frequencies = self.data[:, 0] # frequency always first
            magnitude_data = self.data[:, column * 2 + 1]
            phase_data = self.data[:, column * 2 + 2]

            # scales
            magnitude_scale = match.group(3)
            phase_scale = match.group(4)

            # create data series
            series = ComplexSeries(x=frequencies, magnitude=magnitude_data,
                                   phase=phase_data,
                                   magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            self.add_function(VoltageTransferFunction(series=series,
                                                      source=source,
                                                      sink=sink))

    def _parse_current_components(self, lines):
        """Matches output file current transfer functions

        :param lines: output file lines
        :type lines: Sequence[str]
        """

        # transfer function source is the input
        source = self.parser.input_nodes[0]

        # find transfer functions
        for line in lines:
            match = re.match(self.TF_CURRENT_SINK_REGEX, line)

            if not match:
                continue

            # data column index
            column = int(match.group(1))

            # current sink component
            sink = self.parser.get_component(match.group(3))

            # data
            frequencies = self.data[:, 0] # frequency always first
            magnitude_data = self.data[:, column * 2 + 1]
            phase_data = self.data[:, column * 2 + 2]

            # scales
            magnitude_scale = match.group(4)
            phase_scale = match.group(5)

            # create data series
            series = ComplexSeries(x=frequencies, magnitude=magnitude_data,
                                   phase=phase_data,
                                   magnitude_scale=magnitude_scale,
                                   phase_scale=phase_scale)

            self.add_function(CurrentTransferFunction(series=series,
                                                      source=source,
                                                      sink=sink))

    def _parse_noise_components(self, lines):
        """Matches output file noise spectra

        :param lines: output file lines
        :type lines: Sequence[str]
        """

        # noise sink is the noise node
        sink = self.parser.noise_node

        # find noise component information
        matches = re.search(self.NOISE_VOLTAGE_SOURCE_REGEX, "".join(lines))

        # split into list
        source_strs = matches.group(1).split()

        for index, source_str in enumerate(source_strs, start=1):
            # extract component and noise type
            source_name, noise_type = self._parse_noise_component(source_str)

            # noise source component
            source = self.parser.get_component(source_name)

            frequencies = self.data[:, 0] # frequency always first
            spectrum = self.data[:, index]

            # create data series
            series = Series(x=frequencies, y=spectrum)

            self.add_function(NoiseSpectrum(series=series,
                                            source=source,
                                            sink=sink))#,
                                            #noise_type=noise_type)

    @classmethod
    def _parse_noise_component(cls, source_str):
        # get rid of whitespace around string
        source_str = source_str.strip()

        # look for component name and brackets
        match = re.match(cls.NOISE_COMPONENT_REGEX, source_str)

        component_name = match.group(1)

        # component noise type, e.g. I+ (or empty, for resistors)
        noise_str = match.group(3)

        # group 2 is not empty if noise type is specified
        if match.group(2):
            # op-amp noise; check first character
            if noise_str[0] == "U":
                noise_type = Circuit.NOISE_OPAMP_VOLTAGE
            elif noise_str[0] == "I":
                noise_type = Circuit.NOISE_OPAMP_CURRENT
            else:
                raise ValueError("unrecognised noise type")
        else:
            noise_type = Circuit.NOISE_JOHNSON

        return component_name, noise_type

    @property
    def frequencies(self):
        if not len(self.functions):
            raise Exception("no frequencies available (has LISO been run yet?)")

        # use first function's x-axis
        return self.functions[0].series.x
