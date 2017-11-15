import re

from .circuit import Circuit
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp, Node,
                         Gnd)

class CircuitParser(object):
    COMPONENTS = ["r", "c", "l", "op"]
    DIRECTIVES = ["uinput"]

    def __init__(self):
        self.components = []
        self.nodes = {}
        self.input_node = None
        self.input_impedance = None
        self.loaded = False

    def load(self, filepath):
        """Load a LISO .fil circuit definition file"""

        # open file
        with open(filepath, "r") as obj:
            for tokens in self._fil_tokens(obj.readlines()):
                self._parse_tokens(tokens)

        self.loaded = True

    def circuit(self):
        if not self.loaded:
            raise Exception("File not loaded")

        # create empty circuit
        circuit = Circuit(self.input_node, self.input_impedance)

        # add components
        for component in self.components:
            circuit.add_component(component)

        return circuit

    @staticmethod
    def _fil_tokens(lines) -> str:
        # compile regex expressions
        comment_cleaner = re.compile("#.*?$")

        for line in lines:
            # remove comments and extra whitespace and split into parts
            yield re.sub(comment_cleaner, "", line).split()

    def _parse_tokens(self, tokens) -> None:
        if len(tokens) < 1:
            return

        command = tokens[0]

        if command in self.COMPONENTS:
            # this is a component
            self._parse_component(tokens)
        elif command in self.DIRECTIVES:
            # this is a directive
            self._parse_directive(tokens)

    def _parse_component(self, tokens) -> None:
        component = tokens[0]

        if component == "r":
            obj = self._parse_resistor(tokens)
        elif component == "c":
            obj = self._parse_capacitor(tokens)
        elif component == "l":
            obj = self._parse_inductor(tokens)
        elif component == "op":
            obj = self._parse_opamp(tokens)
        else:
            raise ValueError("Unknown component: %s" % component)

        self.components.append(obj)

    def _parse_lcr(self, _class, tokens):
        name = tokens[1]
        value = tokens[2]
        node1 = self._get_node(tokens[3])
        node2 = self._get_node(tokens[4])

        return _class(name=name, value=value, node1=node1, node2=node2)

    def _parse_resistor(self, *args):
        return self._parse_lcr(Resistor, *args)

    def _parse_capacitor(self, *args):
        return self._parse_lcr(Capacitor, *args)

    def _parse_inductor(self, *args):
        return self._parse_lcr(Inductor, *args)

    def _parse_opamp(self, tokens):
        name = tokens[1]
        model = tokens[2]
        node1 = self._get_node(tokens[3])
        node2 = self._get_node(tokens[4])
        node3 = self._get_node(tokens[5])

        return OpAmp(name=name, model=model, node1=node1, node2=node2,
                     node3=node3)

    def _get_node(self, node_name: str) -> Node:
        node_name = str(node_name)

        if node_name not in self.nodes:
            if node_name.lower() == "gnd":
                node = Gnd()
            else:
                node = Node(node_name)

            self.nodes[node_name] = node

        return self.nodes[node_name]

    def _parse_directive(self, tokens) -> None:
        directive = tokens[0]

        if directive == "uinput":
            if len(tokens) > 3:
                # TODO: handle floating input
                self.input_node_2 = self._get_node(tokens[2])
                self.input_impedance = tokens[3]
            else:
                self.input_node = self._get_node(tokens[1])
                self.input_impedance = tokens[2]
        else:
            raise ValueError("Unknown directive: %s" % directive)
