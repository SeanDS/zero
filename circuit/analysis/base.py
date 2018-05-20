import abc
import copy
import statistics

from ..components import (Component, Resistor, Capacitor, Inductor, OpAmp,
                          Input, Node, ComponentNoise, NodeNoise)

class BaseAnalysis(object, metaclass=abc.ABCMeta):
    """Base class for circuit analysis"""

    def __init__(self, circuit):
        self.circuit = circuit

    def component_index(self, component):
        """Get component serial number.

        Parameters
        ----------
        component : :class:`~.Component`
            component

        Returns
        -------
        :class:`int`
            component serial number

        Raises
        ------
        ValueError
            if component not found
        """

        return self.circuit.components.index(component)

    def node_index(self, node):
        """Get node serial number.

        This does not include the ground node, so the first non-ground node
        has serial number 0.

        Parameters
        ----------
        node : :class:`~.Node`
            node

        Returns
        -------
        :class:`int`
            node serial number

        Raises
        ------
        ValueError
            if ground node is specified or specified node is not found
        """

        if node == Node("gnd"):
            raise ValueError("ground node does not have an index")

        return list(self.circuit.non_gnd_nodes).index(node)

    @property
    def elements(self):
        """Matrix elements.

        Returns a sequence of elements - either components or nodes - in the
        order in which they appear in the matrix

        Yields
        ------
        :class:`~.components.Component`, :class:`~.components.Node`
            matrix elements
        """

        yield from self.circuit.components
        yield from self.circuit.non_gnd_nodes

    @property
    def element_names(self):
        """Names of elements (components and nodes) within the circuit.

        Yields
        ------
        :class:`str`
            matrix element names
        """

        return [element.name for element in self.elements]

    @property
    def mean_resistance(self):
        """Average circuit resistance"""
        return statistics.mean([resistor.resistance for resistor in self.circuit.resistors])