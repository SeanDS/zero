"""Electronic circuit class to which linear components can be added and
on which simulations can be performed."""

import sys
import numpy as np
import logging

HAS_GRAPHVIZ = False
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    pass

from .config import CircuitConfig, OpAmpLibrary
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp,
                         Input, Node)

LOGGER = logging.getLogger("circuit")
CONF = CircuitConfig()
LIBRARY = OpAmpLibrary()

class Circuit(object):
    """Represents an electronic circuit containing linear components.

    A circuit can contain components like :class:`resistors <.Resistor>`,
    :class:`capacitors <.Capacitor>`, :class:`inductors <.Inductor>` and
    :class:`op-amps <.OpAmp>`. These are added to the circuit via
    the :meth:`~Circuit.add_component` method.

    Attributes
    ----------
    components : sequence of :class:`components <.components.Component>`
        The circuit's components.
    nodes : sequence of :class:`nodes <.components.Node>`
        The circuit's nodes.
    prescale : :class:`bool`
        whether to prescale matrix elements into natural units for numerical
        precision purposes
    """

    def __init__(self):
        """Instantiate a new circuit"""

        # empty lists of components and nodes
        self.components = []
        self.nodes = []
        self.prescale = True

    @property
    def non_gnd_nodes(self):
        """Circuit's nodes, excluding ground

        Yields
        ------
        :class:`~Node`
            Circuit nodes, excluding ground.
        """

        return [node for node in self.nodes if node is not Node("gnd")]

    def add_component(self, component):
        """Add component, and its nodes, to the circuit.

        Parameters
        ----------
        component : :class:`~.components.Component`
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

        # register component's nodes
        for node in component.nodes:
            self._add_node(node)

    def add_input(self, *args, **kwargs):
        """Add input to circuit."""
        self.add_component(Input(*args, **kwargs))

    def add_resistor(self, *args, **kwargs):
        """Add resistor to circuit."""
        self.add_component(Resistor(*args, **kwargs))

    def add_capacitor(self, *args, **kwargs):
        """Add capacitor to circuit."""
        self.add_component(Capacitor(*args, **kwargs))

    def add_inductor(self, *args, **kwargs):
        """Add inductor to circuit."""
        self.add_component(Inductor(*args, **kwargs))

    def add_opamp(self, *args, **kwargs):
        """Add op-amp to circuit."""
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

    def _add_node(self, node):
        """Add node to circuit.

        Parameters
        ----------
        node : :class:`~.components.Node`
            node to add

        Raises
        ------
        ValueError
            if the node is None
        """

        if node is None:
            raise ValueError("node cannot be None")

        if node not in self.nodes:
            # add
            self.nodes.append(node)

    def get_component(self, component_name):
        """Get circuit component by name.

        Parameters
        ----------
        component_name : :class:`str`
            name of component to fetch

        Returns
        -------
        :class:`~.components.Component`
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
        """Get circuit node by name.

        Parameters
        ----------
        node_name : :class:`str`
            name of node to fetch

        Returns
        -------
        :class:`~.components.Node`
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
        :class:`int`
            number of components
        """

        return len(self.components)

    @property
    def n_nodes(self):
        """Get number of nodes in circuit

        Returns
        -------
        :class:`int`
            number of nodes
        """

        return len(self.nodes)

    @property
    def resistors(self):
        """Circuit resistors.

        Yields
        ------
        :class:`~.components.Resistor`
            circuit resistors
        """

        return [component for component in self.components
                if isinstance(component, Resistor)]

    @property
    def capacitors(self):
        """Circuit capacitors.

        Yields
        ------
        :class:`~.components.Capacitor`
            circuit capacitors
        """

        return [component for component in self.components
                if isinstance(component, Capacitor)]

    @property
    def inductors(self):
        """Circuit inductors.

        Yields
        ------
        :class:`~.components.Inductor`
            circuit inductors
        """

        return [component for component in self.components
                if isinstance(component, Inductor)]

    @property
    def passive_components(self):
        """Circuit passive components"""

        yield from self.resistors
        yield from self.capacitors
        yield from self.inductors

    @property
    def opamps(self):
        """Circuit op-amps.

        Yields
        ------
        :class:`~.components.OpAmp`
            circuit op-amps
        """

        return [component for component in self.components
                if isinstance(component, OpAmp)]

    @property
    def noise_sources(self):
        """Noise sources in the circuit.

        Yields
        ------
        :class:`~.components.ComponentNoise`
            noise source
        """

        for component in self.components:
            yield from component.noise

    @property
    def input_component(self):
        """Circuit's input component."""
        return self.get_component("input")

    @property
    def input_impedance(self):
        """Circuit's input impedance."""
        return self.input_component.impedance

    @property
    def has_input(self):
        """Check if circuit has an input."""
        try:
            self.input_component
        except ValueError:
            return False

        return True












    def _node_graph(self):
        """Create Graphviz node graph."""

        if not HAS_GRAPHVIZ:
            raise NotImplementedError("Node graph representation requires the "
                                      "graphviz package")

        G = graphviz.Digraph(engine=CONF["graphviz"]["engine"])
        G.attr("node", style=CONF["graphviz"]["node_style"],
               fontname=CONF["graphviz"]["node_font_name"],
               fontsize=CONF["graphviz"]["node_font_size"])
        G.attr("edge", arrowhead=CONF["graphviz"]["edge_arrowhead"])
        G.attr("graph", splines=CONF["graphviz"]["graph_splines"],
               label="Made with graphviz and circuit.py",
               fontname=CONF["graphviz"]["graph_font_name"],
               fontsize=CONF["graphviz"]["graph_font_size"])
        node_map = {}

        def add_connection(C, conn, N):
            if N == 'gnd':
                G.node(C+'_'+N, shape='point', style='invis')
                G.edge(C+'_'+N, C+conn, dir='both', arrowtail='tee',
                       len='0.0', weight='10')
            else:
                if not N in node_map:
                    G.node(N, shape='point', xlabel=N,
                           width='0.1', fillcolor='Red')
                node_map[N] = N
                G.edge(node_map[N], C+conn)

        for C in self.components:
            connections = ['', '']
            if isinstance(C, OpAmp):
                # TODO: move these to components
                attr = {'shape': 'plain', 'margin': '0', 'orientation': '270'}
                attr['label'] = """<<TABLE BORDER="0" BGCOLOR="LightSkyBlue">
                    <TR><TD PORT="plus">+</TD><TD ROWSPAN="3">{0}<BR/>{1}</TD></TR>
                    <TR><TD> </TD></TR>
                    <TR><TD PORT="minus">-</TD></TR>
                </TABLE>>""".format(C.name, C.model)
                connections = [':plus', ':minus', ':e']
            elif isinstance(C, Inductor):
                attr = {'fillcolor': 'MediumSlateBlue', 'shape': 'diamond'}
            elif isinstance(C, Capacitor):
                attr = {'fillcolor': 'YellowGreen', 'shape': 'diamond'}
            elif isinstance(C, Resistor):
                attr = {'fillcolor': 'Orchid', 'shape': 'diamond'}
            elif isinstance(C, Input):
                attr = {'fillcolor': 'Orange',
                        'shape': ['ellipse','box','pentagon'][C.input_type-1]}
            else:
                print('Unrecognised element {0}: {1}'.format(C.name, C.__class__))

            G.node(C.name, **attr)
            for N, connection in zip(C.nodes, connections):
                add_connection(C.name, connection, N.name)

        return G

    if HAS_GRAPHVIZ:
        def _repr_svg_(self):
            """Graphviz rendering for Jupyter notebooks."""
            return self._node_graph()._repr_svg_()
