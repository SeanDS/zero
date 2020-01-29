"""Electronic circuit class to which linear components can be added and
on which simulations can be performed."""

import logging
from copy import deepcopy
import numpy as np

from .config import ZeroConfig, OpAmpLibrary
from .components import (Resistor, Capacitor, Inductor, OpAmp, Node, ElementNotFoundError,
                         ComponentNotFoundError, NodeNotFoundError, NoiseNotFoundError)

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()
LIBRARY = OpAmpLibrary()


class Circuit:
    """Represents an electronic circuit containing components and nodes.

    A circuit can contain components like :class:`resistors <.components.Resistor>`,
    :class:`capacitors <.components.Capacitor>`, :class:`inductors <.components.Inductor>`
    and :class:`op-amps <.components.OpAmp>`. These are added to the circuit via the
    :meth:`add_component` method.

    Attributes
    ----------
    components : :class:`list` of :class:`.Component`
        The circuit components.
    nodes : :class:`set` of :class:`.Node`
        The circuit nodes.
    """
    # Disallowed component names.
    RESERVED_NAMES = ["all", "allop", "allr", "sum"]

    def __init__(self):
        # Empty component list.
        self.components = []
        # Whilst nodes can be generated by looping through the components and building a set,
        # this is quicker at the expense of only a small memory increase.
        self.nodes = set()

    def __getitem__(self, key):
        return self.get_element(key)

    def __contains__(self, key):
        return self.has_element(key)

    def __copy__(self):
        """Copy circuit.

        Creates a new circuit containing the same components as this one. This avoids the problem
        with shallow copying where the `components` and `nodes` fields of this object are retained
        in the new copy and thus changes to the new circuit appear also in this one. This behaviour
        is useful for analyses which copy the circuit before inserting an input component.
        """
        new_instance = self.__class__()
        new_instance.components = list(self.components)
        new_instance.nodes = set(self.nodes)
        return new_instance

    def __deepcopy__(self, memo):
        """Deep copy circuit.

        Creates a new circuit containing new copies of the components in this one. The new
        components have identical properties but are stored in memory as new objects and so any
        references to components from the old circuit in user code will not point to the new
        versions.

        This is useful when you want to create a "master" circuit and modify and analyse several
        independent versions of it.

        Notes
        -----

        :class:`Nodes <Node>` are by design singletons, existing in the global state of the Python
        kernel, and so the new circuit references the same node objects as the old circuit.
        """
        new_instance = self.__class__()
        new_instance.nodes = set()

        # Shallow copy the nodes, and update the memo to make deepcopy think we already deep copied
        # the nodes.
        for node in self.nodes:
            memo[id(node)] = node
            new_instance.nodes.add(node)

        # Deep copy the components. deepcopy() should use the already-copied nodes from above.
        new_instance.components = deepcopy(self.components, memo)

        return new_instance


    @property
    def non_gnd_nodes(self):
        """Circuit nodes, excluding ground.

        Returns
        -------
        :class:`list` of :class:`.Node`
            The circuit nodes.
        """
        return [node for node in self.nodes if node is not Node("gnd")]

    @property
    def elements(self):
        """Circuit nodes and components.

        Yields
        ------
        :class:`.Node`
            The circuit nodes.
        :class:`.Component`
            The circuit components.
        """
        yield from self.non_gnd_nodes
        yield from self.components

    @property
    def opamp_output_nodes(self):
        """Circuit op-amp output nodes.

        Returns
        -------
        :class:`list` of :class:`.Node`
            The op-amp output nodes.
        """
        return [opamp.node3 for opamp in self.opamps]

    def add_component(self, component):
        """Add existing component to circuit.

        Parameters
        ----------
        component : :class:`.Component`
            The component to add.

        Raises
        ------
        ValueError
            If component is None, or already present in the circuit.
        """
        if component is None:
            raise ValueError("component cannot be None")
        if component.name is None:
            # Assign name.
            self._set_default_name(component)
        if component.name in self:
            raise ValueError(f"element with name '{component.name}' already in circuit")
        elif component.name in self.RESERVED_NAMES:
            raise ValueError(f"component name '{component.name}' is reserved")
        # Add component to end of list.
        self.components.append(component)
        # Add nodes.
        for node in component.nodes:
            if node.name in self.component_names:
                raise ValueError(f"node '{node.name}' is the same as existing circuit component")
            self.nodes.add(node)

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

    def add_library_opamp(self, model, **kwargs):
        """Add library op-amp to circuit.

        Keyword arguments can be used to override individual library parameters.

        Parameters
        ----------
        model : :class:`str`
            The op-amp model name.
        """
        # get library data
        data = LIBRARY.get_data(model)

        # override library data with keyword arguments
        data = {**data, **kwargs}

        self.add_opamp(model=OpAmpLibrary.format_name(model), **data)

    def remove_component(self, component):
        """Remove component from circuit.

        Parameters
        ----------
        component : :class:`str` or :class:`.Component`
            The component to remove.

        Raises
        ------
        :class:`.ComponentNotFoundError`
            If the component is not found.
        """
        if isinstance(component, str):
            # Get component by name.
            component = self.get_component(component)
        elif component not in self.components:
            raise ComponentNotFoundError(component)
        self.components.remove(component)
        # Implicitly remove orphaned nodes by regenerating node set from components.
        self._regenerate_node_set()

    def get_component(self, component_name):
        """Get circuit component by name.

        Parameters
        ----------
        component_name : :class:`str` or :class:`.Component`
            The name of the component to fetch.

        Returns
        -------
        :class:`.Component`
            The component.

        Raises
        ------
        :class:`.ComponentNotFoundError`
            If the component is not found.
        """
        # Get the component name from the object, if appropriate.
        component_name = getattr(component_name, "name", component_name)
        name = component_name.lower()
        for component in self.components:
            if name == component.name.lower():
                return component
        raise ComponentNotFoundError(component_name)

    def replace_component(self, current_component, new_component):
        """Replace circuit component with a new one.

        This can be used to replace components of the same type, but can also replace components of
        different types as long as they have the same number of nodes.

        Parameters
        ----------
        current_component : :class:`.Component`
            The component to replace.
        new_component : :class:`str` or :class:`.Component`
            The new component.

        Raises
        ------
        :class:`.ComponentNotFoundError`
            If the current component is not in the circuit.
        ValueError
            If the new component is already in the circuit, or if the nodes of the new component are
            incompatible with those of the current component.
        """
        current_component = self.get_component(current_component)
        if new_component in self.components:
            raise ValueError(f"{new_component} is already in the circuit")
        if len(current_component.nodes) != len(new_component.nodes):
            raise ValueError(f"{current_component} and {new_component} nodes are incompatible")
        # Copy the nodes.
        nodes = current_component.nodes
        # Do the replacement.
        self.remove_component(current_component)
        LOGGER.debug(f"Overwriting {new_component}'s nodes with those from {current_component}'")
        new_component.nodes = nodes
        self.add_component(new_component)

    def has_component(self, component_name):
        """Check if component is present in circuit.

        Parameters
        ----------
        component_name : str
            The name of the component to check.

        Returns
        -------
        :class:`bool`
            True if component exists, False otherwise.
        """
        try:
            self.get_component(component_name)
        except ComponentNotFoundError:
            return False
        return True

    def get_node(self, node_name):
        """Get circuit node by name.

        Parameters
        ----------
        node_name : :class:`str` or :class:`.Node`
            The name of the node to fetch.

        Returns
        -------
        :class:`.Node`
            The node.

        Raises
        ------
        :class:`NodeNotFoundError`
            If the node is not found.
        """
        # Get the node name from the object, if appropriate.
        node_name = getattr(node_name, "name", node_name)
        name = node_name.lower()
        for node in self.nodes:
            if name == node.name.lower():
                return node
        raise NodeNotFoundError(name)

    def has_node(self, node_name):
        """Check if node is present in circuit.

        Parameters
        ----------
        node_name : str
            The name of the node to check.

        Returns
        -------
        :class:`bool`
            True if node exists, False otherwise.
        """
        try:
            self.get_node(node_name)
        except NodeNotFoundError:
            return False
        return True

    def get_element(self, element_name):
        """Get circuit element (component or node) by name.

        Parameters
        ----------
        element_name : :class:`str`
            The name of the element to fetch.

        Returns
        -------
        :class:`.Component` or :class:`.Node`
            The found component or element.

        Raises
        ------
        :class:`ElementNotFoundError`
            If the element is not found.
        """
        name = element_name.lower()
        try:
            return self.get_component(name)
        except ComponentNotFoundError:
            pass
        try:
            return self.get_node(name)
        except NodeNotFoundError:
            pass
        raise ElementNotFoundError(element_name)

    def has_element(self, element_name):
        """Check if element (component or node) is present in circuit.

        Parameters
        ----------
        element_name : str
            The name of the element to check.

        Returns
        -------
        :class:`bool`
            True if element exists, False otherwise.
        """
        return self.has_component(element_name) or self.has_node(element_name)

    def get_noise(self, noise_name):
        """Get noise by component or node name.

        Parameters
        ----------
        noise_name : :class:`str`
            The name of the noise to fetch.

        Returns
        -------
        :class:`Noise`
            The noise.

        Raises
        ------
        :class:`NoiseNotFoundError`
            If the noise is not found.
        """
        name = noise_name.lower()

        for noise in self.noise_sources:
            if name == noise.label.lower():
                return noise

        raise NoiseNotFoundError(name)

    def set_inductor_coupling(self, inductor_1, inductor_2, coupling_factor=1):
        """Set the coupling factor between the specified inductors

        Parameters
        ----------
        inductor_1, inductor_2 : :class:`str` or :class:`.components.Inductor`
            The inductors to couple.
        coupling_factor : any, optional
            The coupling factor between the specified inductors, specified between 0 and 1. A
            coupling factor less than 1 represents loss.

        Raises
        ------
        :class:`ValueError`
            If a specified inductor is not an inductor.
        """
        # get inductors by name
        inductor_1 = self.get_component(inductor_1)
        inductor_2 = self.get_component(inductor_2)

        if not isinstance(inductor_1, Inductor):
            raise TypeError(f"component '{inductor_1}' is not an inductor")
        if not isinstance(inductor_2, Inductor):
            raise TypeError(f"component '{inductor_2}' is not an inductor")

        # check if we are overwriting something
        if inductor_1 in inductor_2.coupled_inductors or inductor_2 in inductor_1.coupled_inductors:
            LOGGER.warning("overwriting coupling factor between '%s' and '%s'",
                           inductor_1, inductor_2)

        # set coupling factors
        inductor_1.coupling_factors[inductor_2] = coupling_factor
        inductor_2.coupling_factors[inductor_1] = coupling_factor

    def _set_default_name(self, component):
        """Set a default name unique to this circuit for the specified component

        Parameters
        ----------
        component : :class:`Component`
            The component to set the name for.
        """
        # base name of component
        base = component.BASE_NAME

        # number to append
        count = 1

        # first attempt
        new_name = f"{base}{count}"

        while new_name in self.component_names:
            # next attempt
            count += 1
            new_name = f"{base}{count}"

        # set new name
        component.name = new_name

        # set flag showing component has been automatically named
        component.autonamed = True

        LOGGER.info("component %s assigned name %s", component, component.name)

    def _regenerate_node_set(self):
        """Regenerate nodes from components.

        This is used for example when a component is removed, and its nodes may or may not be
        shared by other components. To keep the set of nodes in sync, this method regenerates
        the set using the present components.
        """
        LOGGER.debug("regenerating node set from components present in circuit")
        self.nodes = set([node for component in self.components for node in component.nodes])

    @property
    def component_names(self):
        """The names of the components in the circuit.

        Returns
        -------
        :class:`list` of :class:`str`
            The component names.
        """
        return [component.name for component in self.components]

    @property
    def node_names(self):
        """The names of the nodes in the circuit.

        Returns
        -------
        :class:`list` of :class:`str`
            The node names.
        """
        return [node.name for node in self.nodes]

    @property
    def element_names(self):
        """The names of the elements (components and nodes) in the circuit.

        Yields
        ------
        :class:`str`
            The element names.
        """
        yield from self.component_names
        yield from self.node_names

    @property
    def n_components(self):
        """The number of components in the circuit.

        Returns
        -------
        :class:`int`
            The number of components.
        """
        return len(self.components)

    @property
    def n_nodes(self):
        """The number of nodes in the circuit.

        Returns
        -------
        :class:`int`
            The number of nodes.
        """
        return len(self.nodes)

    @property
    def resistors(self):
        """The circuit resistors.

        Returns
        -------
        :class:`list` of :class:`.Resistor`
            The resistors in the circuit.
        """
        return [component for component in self.components if component.element_type == "resistor"]

    @property
    def capacitors(self):
        """The circuit capacitors.

        Returns
        -------
        :class:`list` of :class:`.Capacitor`
            The capacitors in the circuit.
        """
        return [component for component in self.components if component.element_type == "capacitor"]

    @property
    def inductors(self):
        """The circuit inductors.

        Returns
        -------
        :class:`list` of :class:`.Inductor`
            The inductors in the circuit.
        """
        return [component for component in self.components if component.element_type == "inductor"]

    @property
    def passive_components(self):
        """The circuit passive components.

        Yields
        ------
        :class:`.Resistor`
            The resistors in the circuit.
        :class:`.Capacitor`
            The capacitors in the circuit.
        :class:`.Inductor`
            The inductors in the circuit.
        """
        yield from self.resistors
        yield from self.capacitors
        yield from self.inductors

    @property
    def opamps(self):
        """The op-amps in the circuit.

        Returns
        -------
        :class:`list` of :class:`.OpAmp`
            The op-amps.
        """
        return [component for component in self.components if component.element_type == "op-amp"]

    @property
    def noise_sources(self):
        """The noise sources in the circuit.

        Returns
        -------
        :class:`list` of :class:`.ComponentNoise` or :class:`.NodeNoise`
            The component and node noise sources.
        """
        return [noise for component in self.components for noise in component.noise]

    @property
    def resistor_noise_sources(self):
        """The resistor noise sources in the circuit.

        Yields
        ------
        :class:`.Noise`
            The resistor noise source.
        """
        for resistor in self.resistors:
            yield from resistor.noise

    @property
    def opamp_noise_sources(self):
        """The op-amp noise sources in the circuit.

        Yields
        ------
        :class:`.Noise`
            The op-amp noise source.
        """
        for opamp in self.opamps:
            yield from opamp.noise

    @property
    def input_component(self):
        """The circuit input component.

        Returns
        -------
        :class:`.Input`
            The circuit input.
        """
        return self.get_component("input")

    @property
    def input_impedance(self):
        """The circuit input impedance.

        Returns
        -------
        :class:`float`
            circuit input impedance
        """
        return self.input_component.impedance

    @property
    def has_input(self):
        """Whether the circuit has an input.

        Returns
        -------
        :class:`bool`
            True if the circuit has an input, False otherwise.
        """
        try:
            self.input_component
        except ComponentNotFoundError:
            return False

        return True

    def __repr__(self):
        """Circuit text representation"""
        if self.n_components == 1:
            cmp_str = "component"
        else:
            cmp_str = "components"

        if self.n_nodes == 1:
            node_str = "node"
        else:
            node_str = "nodes"

        text = f"Circuit with {self.n_components} {cmp_str} and {self.n_nodes} {node_str}"

        if self.n_components > 0:
            text += "\n"

            # field size
            iw = int(np.ceil(np.log10(len(self.components))))

            # components ordered alphabetically
            ordered = sorted(self.components, key=lambda cmp: (cmp.__class__.__name__, cmp.name))

            for index, component in enumerate(ordered, start=1):
                text += f"\n\t{index:{iw}}. {component}"

        return text
