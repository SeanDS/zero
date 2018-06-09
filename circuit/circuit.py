"""Electronic circuit class to which linear components can be added and
on which simulations can be performed."""

import sys
import numpy as np
import logging

from .config import CircuitConfig, OpAmpLibrary
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp,
                         Input, Node)
from .display import NodeGraph

LOGGER = logging.getLogger("circuit")
CONF = CircuitConfig()
LIBRARY = OpAmpLibrary()

class Circuit(object):
    """Represents an electronic circuit containing linear components

    A circuit can contain components like :class:`resistors <.components.Resistor>`,
    :class:`capacitors <.components.Capacitor>`, :class:`inductors <.components.Inductor>`
    and :class:`op-amps <.components.OpAmp>`. These are added to the circuit via the
    :meth:`add_component` method.

    Attributes
    ----------
    components : :class:`list` of :class:`.Component`
        circuit components
    prescale : :class:`bool`
        whether to prescale matrix elements into natural units for numerical
        precision purposes
    """

    def __init__(self):
        # empty lists of components and nodes
        self.components = []
        self.prescale = True

    @property
    def nodes(self):
        """Circuit's nodes, including ground, if present
        
        Returns
        -------
        nodes : :class:`set` of :class:`.Node`
            circuit nodes
        """
        return set([node for component in self.components for node in component.nodes])

    @property
    def non_gnd_nodes(self):
        """Circuit nodes, excluding ground

        Returns
        -------
        nodes : :class:`list` of :class:`.Node`
            circuit nodes
        """
        return [node for node in self.nodes if node is not Node("gnd")]

    @property
    def elements(self):
        """Circuit nodes and components
        
        Yields
        ------
        node: :class:`.Node`
            nodes
        component : :class:`.Component`
            components
        """
        yield from self.non_gnd_nodes
        yield from self.components

    @property
    def opamp_output_nodes(self):
        """Op-amp output nodes

        Returns
        -------
        nodes : :class:`list` of :class:`.Node`
            op-amp output nodes
        """
        return [opamp.node3 for opamp in self.opamps]

    def add_component(self, component):
        """Add circuit component

        Parameters
        ----------
        component : :class:`.Component`
            component to add

        Raises
        ------
        ValueError
            if component is None, or already present in the circuit
        """

        if component is None:
            raise ValueError("component cannot be None")
        elif component in self.components:
            raise ValueError("component %s already in circuit" % component)

        # add component to end of list
        self.components.append(component)

    def add_input(self, *args, **kwargs):
        """Add input to circuit"""
        self.add_component(Input(*args, **kwargs))

    def add_resistor(self, *args, **kwargs):
        """Add resistor to circuit"""
        self.add_component(Resistor(*args, **kwargs))

    def add_capacitor(self, *args, **kwargs):
        """Add capacitor to circuit"""
        self.add_component(Capacitor(*args, **kwargs))

    def add_inductor(self, *args, **kwargs):
        """Add inductor to circuit"""
        self.add_component(Inductor(*args, **kwargs))

    def add_opamp(self, *args, **kwargs):
        """Add op-amp to circuit"""
        self.add_component(OpAmp(*args, **kwargs))

    def add_library_opamp(self, model, *args, **kwargs):
        """Add library op-amp to circuit
        
        Keyword arguments can be used to override individual library parameters.
        """
        # get library data
        data = LIBRARY.get_data(model)

        # override library data with keyword arguments
        data = {**data, **kwargs}

        self.add_opamp(model=OpAmpLibrary.format_name(model), *args, **data)

    def remove_component(self, component):
        """Remove circuit component

        Parameters
        ----------
        component : :class:`str` or :class:`.Component`
            component to remove
        
        Raises
        ------
        ValueError
            if the component is not in the circuit
        """
        # remove
        self.components.remove(component)

    def get_component(self, component_name):
        """Get circuit component by name

        Parameters
        ----------
        component_name : str
            name of component to fetch

        Returns
        -------
        :class:`.Component`
            component

        Raises
        ------
        ValueError
            if component not found
        """
        component_name = component_name.lower()

        for component in self.components:
            if component.name.lower() == component_name:
                return component

        raise ValueError("component %s not found" % component_name)

    def get_node(self, node_name):
        """Get circuit node by name

        Parameters
        ----------
        node_name : :class:`str`
            name of node to fetch

        Returns
        -------
        node : :class:`.Node`
            node

        Raises
        ------
        ValueError
            if node not found
        """
        node_name = node_name.lower()

        for node in self.nodes:
            if node.name.lower() == node_name:
                return node

        raise ValueError("node %s not found" % node_name)

    @property
    def n_components(self):
        """Get number of components in circuit

        Returns
        -------
        int
            number of components
        """
        return len(self.components)

    @property
    def n_nodes(self):
        """Get number of nodes in circuit

        Returns
        -------
        int
            number of nodes
        """
        return len(self.nodes)

    @property
    def resistors(self):
        """Circuit resistors

        Returns
        -------
        resistors : :class:`list` of :class:`.Resistor`
            circuit resistors
        """
        return [component for component in self.components if isinstance(component, Resistor)]

    @property
    def capacitors(self):
        """Circuit capacitors

        Returns
        -------
        capacitors : :class:`list` of :class:`.Capacitor`
            circuit capacitors
        """
        return [component for component in self.components if isinstance(component, Capacitor)]

    @property
    def inductors(self):
        """Circuit inductors

        Returns
        -------
        inductors : :class:`list` of :class:`.Inductor`
            circuit inductors
        """
        return [component for component in self.components if isinstance(component, Inductor)]

    @property
    def passive_components(self):
        """Circuit passive components
        
        Yields
        ------
        resistor : :class:`.Resistor`
            circuit resistors
        capacitor : :class:`.Capacitor`
            circuit capacitors
        inductor : :class:`.Inductor`
            circuit inductors
        """
        yield from self.resistors
        yield from self.capacitors
        yield from self.inductors

    @property
    def opamps(self):
        """Circuit op-amps

        Returns
        -------
        opamps : :class:`list` of :class:`.OpAmp`
            circuit op-amps
        """
        return [component for component in self.components if isinstance(component, OpAmp)]

    @property
    def noise_sources(self):
        """Noise sources in the circuit

        Returns
        -------
        noise_sources : :class:`list` of :class:`.ComponentNoise`
            circuit component noise sources
        """
        return [noise for component in self.components for noise in component.noise]

    @property
    def input_component(self):
        """Circuit input component"""
        return self.get_component("input")

    @property
    def input_impedance(self):
        """Circuit input impedance"""
        return self.input_component.impedance

    @property
    def has_input(self):
        """Check if circuit has an input"""
        try:
            self.input_component
        except ValueError:
            return False

        return True
    
    def __repr__(self):
        """Circuit text representation"""
        if self.n_components > 1:
            cmp_str = "components"
        else:
            cmp_str = "component"

        if self.n_nodes > 1:
            node_str = "nodes"
        else:
            node_str = "node"

        text = "Circuit with {n_cmps} {cmp_str} and {n_nodes} {node_str}".format(
            n_cmps=self.n_components, cmp_str=cmp_str, n_nodes=self.n_nodes, node_str=node_str)
        
        if self.n_components > 0:
            text += "\n"

            # field size
            iw = int(np.ceil(np.log10(len(self.components))))

            for index, component in enumerate(sorted(self.components, key=lambda cmp: (cmp.__class__.__name__, cmp.name)), start=1):
                text += "\n\t{index:{iw}}. {cmp}".format(index=index, iw=iw, cmp=component)

        return text