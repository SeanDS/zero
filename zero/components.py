"""Electronic components"""

import abc
from collections.abc import MutableMapping
import numpy as np

from .elements import BaseElement, ElementNotFoundError
from .noise import OpAmpVoltageNoise, OpAmpCurrentNoise, ResistorJohnsonNoise, NoiseNotFoundError
from .misc import NamedInstance
from .format import Quantity
from .config import ZeroConfig, LibraryOpAmp

CONF = ZeroConfig()


class Component(BaseElement, metaclass=abc.ABCMeta):
    """Represents a circuit component.

    Parameters
    ----------
    name : :class:`str`, optional
        The component name. Must be unique.
    nodes : sequence of :class:`~Node` or :class:`str`, optional
        The component nodes.

    Attributes
    ----------
    noise : :class:`set` of :class:`.ComponentNoise`
        The component noise sources.
    """
    ELEMENT_TYPE = "component"
    ELEMENT_UNIT = "A"
    BASE_NAME = "?"
    DISPLAY_UNIT = "?"

    def __init__(self, name=None, nodes=None):
        super().__init__()
        if name is not None:
            name = str(name)
        if nodes is None:
            nodes = []
        # Defaults.
        self._nodes = []
        self.noise = []
        self.autonamed = False

        self.name = name
        self.nodes = nodes

    @property
    def nodes(self):
        """The component nodes.

        Returns
        -------
        :class:`list` of :class:`~Node`
            The component nodes.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        nodes = list(nodes)
        for index, node in enumerate(nodes):
            if not isinstance(node, Node):
                # Parse as a node name.
                nodes[index] = Node(str(node))
        self._nodes = nodes

    def add_noise(self, noise):
        """Add a noise source to the component.

        Parameters
        ----------
        noise : :class:`~Noise`
            The noise to add.

        Raises
        ------
        ValueError
            If specified noise is already present.
        """
        noise.component = self
        if noise in self.noise:
            raise ValueError(f"specified noise '{noise}' already exists in '{self}'")
        self.noise.append(noise)

    @property
    def label(self):
        """Get component label.

        Returns
        -------
        :class:`str`
            The component label.
        """
        name = self.name

        if name is None:
            # Name not set.
            name = f"{self.__class__.__name__} (no name)"

        return name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return self.name == getattr(other, "name", None)

    def __hash__(self):
        """Components uniquely defined by their name"""
        return hash((self.name))


class PassiveComponent(Component, metaclass=abc.ABCMeta):
    """Represents a passive component.

    A passive component is one that consumes or temporarily stores energy, but
    does not produce or amplify it. Examples include
    :class:`resistors <Resistor>`, :class:`capacitors <Capacitor>` and
    :class:`inductors <Inductor>`.

    Parameters
    ----------
    value : any, optional
        The component value.
    node1 : :class:`~Node`, optional
        The first component node.
    node2 : :class:`~Node`, optional
        The second component node.

    Attributes
    ----------
    value : :class:`float`
        The component value.
    """
    def __init__(self, value=None, node1=None, node2=None, *args, **kwargs):
        super().__init__(nodes=[node1, node2], *args, **kwargs)
        self.value = value

    @property
    def value(self):
        """The component value.

        Returns
        -------
        :class:`float`
            The component value.
        """
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            value = Quantity(value, self.DISPLAY_UNIT)

        self._value = value

    @property
    def node1(self):
        """The first component node.

        Returns
        -------
        :class:`.Node`
            The first component node.
        """
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = Node(node)

    @property
    def node2(self):
        """The second component node.

        Returns
        -------
        :class:`.Node`
            The second component node.
        """
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = Node(node)

    @abc.abstractmethod
    def impedance(self, frequency):
        """The passive impedance."""
        return NotImplemented


class OpAmp(LibraryOpAmp, Component):
    """Represents an (almost) ideal op-amp.

    An op-amp produces :class:`voltage noise <VoltageNoise>` across its input
    and output :class:`nodes <Node>`, and :class:`current noise <CurrentNoise>`
    is present at its input :class:`nodes <Node>`.

    Parameters
    ----------
    node1 : :class:`Node`
        Non-inverting input node.
    node2 : :class:`Node`
        Inverting input node.
    node3 : :class:`Node`
        Output node.
    """
    ELEMENT_TYPE = "op-amp"
    BASE_NAME = "op"

    def __init__(self, node1, node2, node3, **kwargs):
        super().__init__(nodes=[node1, node2, node3], **kwargs)
        # Op-amp voltage noise.
        self.add_noise(OpAmpVoltageNoise(component=self))
        # Op-amp input current noise.
        if self.node1 is not Node("gnd"):
            # Non-inverting input noise.
            self.add_noise(OpAmpCurrentNoise(node=self.node1))
        if self.node2 is not Node("gnd"):
            # Inverting input noise.
            self.add_noise(OpAmpCurrentNoise(node=self.node2))

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = Node(node)

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = Node(node)

    @property
    def node3(self):
        return self.nodes[2]

    @node3.setter
    def node3(self, node):
        self.nodes[2] = Node(node)

    @property
    def has_voltage_noise(self):
        return "voltage" in [noise.noise_type for noise in self.noise]

    @property
    def has_non_inv_current_noise(self):
        return "current" in [noise.noise_type for noise in self.noise
                             if hasattr(noise, "node") and noise.node == self.node1]

    @property
    def has_inv_current_noise(self):
        return "current" in [noise.noise_type for noise in self.noise
                             if hasattr(noise, "node") and noise.node == self.node2]

    @property
    def voltage_noise(self):
        for noise in self.noise:
            if noise.noise_type == "voltage":
                return noise
        raise NoiseNotFoundError("voltage noise")

    @property
    def non_inv_current_noise(self):
        for noise in self.noise:
            if noise.noise_type == "current":
                if noise.node == self.node1:
                    return noise
        raise NoiseNotFoundError("non-inverting current noise")

    @property
    def inv_current_noise(self):
        for noise in self.noise:
            if noise.noise_type == "current":
                if noise.node == self.node2:
                    return noise
        raise NoiseNotFoundError("inverting current noise")

    def __str__(self):
        suffix = f" [in+={self.node1}, in-={self.node2}, out={self.node3}, model={self.model}]"
        return Component.__str__(self) + suffix


class Input(Component):
    """Represents the circuit's voltage input"""
    ELEMENT_TYPE = "input"
    BASE_NAME = "in"

    def __init__(self, nodes, input_type, impedance=None, is_noise=False, **kwargs):
        self._impedance = None
        self.input_type = input_type
        self.is_noise = bool(is_noise)
        self.impedance = impedance
        super().__init__(name="input", nodes=nodes, **kwargs)

    @property
    def impedance(self):
        return self._impedance

    @impedance.setter
    def impedance(self, impedance):
        if impedance is None:
            return
        self._impedance = Quantity(impedance, "Ω")

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = Node(node)

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = Node(node)

    @property
    def node_n(self):
        return self.node1

    @node_n.setter
    def node_n(self, node):
        self.node1 = node

    @property
    def node_p(self):
        return self.node2

    @node_p.setter
    def node_p(self, node):
        self.node2 = node

    def __str__(self):
        if self.impedance:
            z = self.impedance
        else:
            z = "default"
        return super().__str__() + f" [in={self.node1}, out={self.node2}, Z={z}]"


class Resistor(PassiveComponent):
    """Represents a resistor or set of series or parallel resistors"""
    ELEMENT_TYPE = "resistor"
    DISPLAY_UNIT = "Ω"
    BASE_NAME = "r"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register Johnson noise.
        self.add_noise(ResistorJohnsonNoise())

    @property
    def resistance(self):
        """Resistance in ohms."""
        return self.value

    @resistance.setter
    def resistance(self, resistance):
        self.value = resistance

    def impedance(self, *args):
        """The impedance.

        Returns
        -------
        :class:`complex`
            The impedance.
        """
        return self.resistance

    @property
    def johnson_noise(self):
        for noise in self.noise:
            if noise.noise_type == "johnson":
                return noise

        raise ValueError("no Johnson noise")

    def __str__(self):
        return super().__str__() + f" [in={self.node1}, out={self.node2}, R={self.resistance}]"


class Capacitor(PassiveComponent):
    """Represents a capacitor or set of series or parallel capacitors"""
    ELEMENT_TYPE = "capacitor"
    DISPLAY_UNIT = "F"
    BASE_NAME = "c"

    @property
    def capacitance(self):
        """Capacitance in farads."""
        return self.value

    @capacitance.setter
    def capacitance(self, capacitance):
        self.value = capacitance

    def impedance(self, frequency):
        """The impedance.

        Parameters
        ----------
        frequency : :class:`float` or array_like

        Returns
        -------
        :class:`complex`
            The impedance.
        """
        return 1 / (2j * np.pi * frequency * self.capacitance)

    def __str__(self):
        return super().__str__() + f" [in={self.node1}, out={self.node2}, C={self.capacitance}]"


class Inductor(PassiveComponent):
    """Represents an inductor or set of series or parallel inductors"""
    ELEMENT_TYPE = "inductor"
    DISPLAY_UNIT = "H"
    BASE_NAME = "l"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # default inductor coupling factor map
        self.coupling_factors = CouplingFactorDict(self)

    @property
    def inductance(self):
        """Inductance in henries."""
        return self.value

    @inductance.setter
    def inductance(self, inductance):
        self.value = inductance

    def impedance(self, frequency):
        """The impedance.

        Parameters
        ----------
        frequency : :class:`float` or array_like
            The frequency.

        Returns
        -------
        :class:`complex`
            The impedance.
        """
        return 2j * np.pi * frequency * self.inductance

    def inductance_from(self, other):
        """Calculate the mutual inductance this inductor has with the specified inductor

        Parameters
        ----------
        other : :class:`Inductor`
            The other inductor.

        Returns
        -------
        :class:`.Quantity`
            The mutual inductance between this inductor and the specified one.

        Raises
        ------
        :class:`TypeError`
            If the specified inductor is not of type :class:`Inductor`
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"specified component '{other}' is not an inductor")

        coupling_factor = self.coupling_factors[other]
        mutual_inductance = coupling_factor * np.sqrt(self.inductance * other.inductance)

        return Quantity(mutual_inductance, units=self.DISPLAY_UNIT)

    def impedance_from(self, other, frequency):
        """Calculate the impedance this inductor has due to the specified coupled inductor

        Parameters
        ----------
        other : :class:`Inductor`
            The other inductor.
        frequency : :class:`float` or array_like
            The frequency.

        Returns
        -------
        :class:`complex`
            The impedance.
        """
        return 2j * np.pi * frequency * self.inductance_from(other)

    @property
    def coupled_inductors(self):
        """Inductors coupled to this one"""
        return self.coupling_factors.keys()

    def __str__(self):
        return super().__str__() + f" [in={self.node1}, out={self.node2}, L={self.inductance}]"


class ComponentNotFoundError(ElementNotFoundError):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, message="component '%s' not found", *args, **kwargs)


class Node(BaseElement, metaclass=NamedInstance):
    """Represents a circuit node (connection between components)

    Nodes are considered equal if they have the same case-independent name. Nodes are singletons,
    and as such instantiating a node with a name matching that of a previously instantiated node
    will result in the previous object being returned.

    Parameters
    ----------
    name : :class:`str`
        Node name.
    """
    ELEMENT_UNIT = "V"

    def __init__(self, name):
        """Instantiate a new node."""
        super().__init__()
        self.name = str(name)

    @property
    def label(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class NodeNotFoundError(ElementNotFoundError):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, message="node '%s' not found", *args, **kwargs)


class CouplingFactorDict(MutableMapping):
    """Collection to get and set coupling factors between inductors"""
    def __init__(self, inductor, *args, **kwargs):
        self.inductor = inductor

        # create dict to store things
        self._couplings = dict()

        # initialise data
        self.update(dict(*args, **kwargs))

    def __getitem__(self, inductor):
        """Get coupling factor for specified inductor

        If there is no coupling factor defined between the inductors, it is assumed to be zero.

        Parameters
        ----------
        inductor : :class:`.Inductor`
            The inductor to get the coupling for.

        Returns
        -------
        :class:`.Quantity`
            The coupling factor.

        Raises
        ------
        :class:`TypeError`
            If the specified component is not an inductor.
        """
        if not isinstance(inductor, Inductor):
            raise TypeError(f"specified component, '{inductor}', is not an inductor")

        return self._couplings.get(inductor, 0)

    def __setitem__(self, inductor, coupling_factor):
        """Set coupling factor for specified inductor

        Parameters
        ----------
        inductor : :class:`.Inductor`
            The inductor to couple to the inductor contained within this.
        coupling_factor : any
            The coupling factor to use.

        Raises
        ------
        :class:`TypeError`
            If the specified component is not an inductor.
        :class:`ValueError`
            If the specified coupling factor is outside the range [0, 1].
        """
        if not isinstance(inductor, Inductor):
            raise TypeError(f"specified component, '{inductor}', is not an inductor")

        # parse value
        coupling_factor = Quantity(coupling_factor)

        if coupling_factor < 0 or coupling_factor > 1:
            raise ValueError("specified coupling factor must be between 0 and 1")

        self._couplings[inductor] = coupling_factor

    def __delitem__(self, key):
        del self._couplings[key]

    def __iter__(self):
        return iter(self._couplings)

    def __len__(self):
        return len(self._couplings)

    def __contains__(self, key):
        return key in self._couplings
