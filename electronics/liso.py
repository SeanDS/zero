import re

from .circuit import Circuit
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp, Node,
                         Gnd)
from typing import Generator, Callable, Sequence, List, Dict

class CircuitParser(object):
    COMPONENTS = ["r", "c", "l", "op"]
    DIRECTIVES = ["uinput"]

    COMMENT_REGEX = re.compile("#.*?$")

    def __init__(self):
        # default values
        self.components = []
        self.nodes = {}
        self.input_node = None
        self.input_impedance = 0
        self.loaded = False

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
        """Get circuit object representing LISO model

        :return: circuit object
        :rtype: :class:`~Circuit`
        """

        if not self.loaded:
            raise Exception("File not loaded")

        # create empty circuit
        circuit = Circuit(self.input_node, self.input_impedance)

        # add components
        for component in self.components:
            circuit.add_component(component)

        return circuit

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
        elif command in self.DIRECTIVES:
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
        """

        if directive in ["uinput", "vinput"]:
            if len(options) > 3:
                # TODO: handle floating input
                self.input_node_2 = self._get_node(options[1])
                self.input_impedance = int(options[2])
            else:
                self.input_node = self._get_node(options[0])
                self.input_impedance = int(options[1])
        else:
            raise ValueError("Unknown directive: %s" % directive)
