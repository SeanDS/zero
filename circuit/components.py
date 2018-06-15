"""Electronic components"""

import abc
import logging
import numpy as np

from .misc import NamedInstance
from .format import SIFormatter
from .config import CircuitConfig

CONF = CircuitConfig()

class Component(object, metaclass=abc.ABCMeta):
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

    TYPE = "component"

    def __init__(self, name=None, nodes=None):
        if name is not None:
            name = str(name)

        if nodes is None:
            nodes = []

        # defaults
        self._nodes = []
        self.noise = set()

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
        for node in list(nodes):
            if not isinstance(node, Node):
                # parse node name
                node = Node(str(node))

            self._nodes.append(node)

    def add_noise(self, noise):
        """Add a noise source to the component.

        Parameters
        ----------
        noise : :class:`~Noise`
            The noise to add.
        """
        self.noise.add(noise)

    def label(self):
        """Get component label.

        Returns
        -------
        :class:`str`
            The component label.
        """
        return self.name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.label()
    
    def __eq__(self, other):
        return self.name == other.name
    
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

    UNIT = "?"

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
            value, _ = SIFormatter.parse(value)

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
        self.nodes[0] = node

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
        self.nodes[1] = node

    @abc.abstractmethod
    def impedance(self, frequency):
        """The passive impedance."""
        return NotImplemented

class OpAmp(Component):
    """Represents an (almost) ideal op-amp.

    An op-amp produces :class:`voltage noise <VoltageNoise>` across its input
    and output :class:`nodes <Node>`, and :class:`current noise <CurrentNoise>`
    is present at its input :class:`nodes <Node>`.

    The default parameter values are those used in LISO, which are
    themselves based on the OP27 PMI (information provided by Gerhard
    Heinzel).

    Parameters
    ----------
    model : :class:`str`
        Model name.
    node1 : :class:`Node`
        Non-inverting input node.
    node2 : :class:`Node`
        Inverting input node.
    node3 : :class:`Node`
        Output node.
    a0 : :class:`float`, optional
        Open loop gain.
    gbw : :class:`float`, optional
        Gain-bandwidth product.
    delay : :class:`float`, optional
        Delay.
    zeros : sequence, optional
        Zeros.
    poles : sequence, optional
        Poles.
    v_noise : :class:`float`, optional
        Flat voltage noise.
    i_noise : :class:`float`, optional
        Float current noise.
    v_corner : :class:`float`, optional
        Voltage noise corner frequency.
    i_corner : :class:`float`, optional
        Current noise corner frequency.
    v_max : :class:`float`, optional
        Maximum input voltage.
    i_max : :class:`float`, optional
        Maximum output current.
    slew_rate : :class:`float`, optional
        Slew rate.
    """

    TYPE = "op-amp"

    def __init__(self, model, node1, node2, node3, a0=1.5e6, gbw=8e6,
                 delay=0, zeros=np.array([]), poles=np.array([]),
                 v_noise=3.2e-9, i_noise=0.4e-12, v_corner=2.7, i_corner=140,
                 v_max=12, i_max=0.06, slew_rate=1e6, *args, **kwargs):
        # call parent constructor
        super().__init__(nodes=[node1, node2, node3], *args, **kwargs)

        # default properties
        self.params = {"a0": SIFormatter.parse(a0)[0], # gain
                       "gbw": SIFormatter.parse(gbw)[0], # gain-bandwidth product (Hz)
                       "delay": SIFormatter.parse(delay)[0], # delay (s)
                       "zeros": np.array(zeros), # array of additional zeros
                       "poles": np.array(poles), # array of additional poles
                       "vn": SIFormatter.parse(v_noise)[0], # voltage noise (V/sqrt(Hz))
                       "in": SIFormatter.parse(i_noise)[0], # current noise (A/sqrt(Hz))
                       "vc": SIFormatter.parse(v_corner)[0], # voltage noise corner frequency (Hz)
                       "ic": SIFormatter.parse(i_corner)[0], # current noise corner frequency (Hz)
                       "vmax": SIFormatter.parse(v_max)[0], # maximum output voltage amplitude (V)
                       "imax": SIFormatter.parse(i_max)[0], # maximum output current amplitude (A)
                       "sr": SIFormatter.parse(slew_rate)[0]} # maximum slew rate (V/s)

        # set model name
        self.model = model

        # op-amp voltage noise
        self.add_noise(VoltageNoise(component=self,
                                    function=self._noise_voltage))

        # op-amp input current noise
        if self.node1 is not Node("gnd"):
            # non-inverting input noise
            self.add_noise(CurrentNoise(node=self.node1, component=self,
                                        function=self._noise_current))
        if self.node2 is not Node("gnd"):
            # inverting input noise
            self.add_noise(CurrentNoise(node=self.node2, component=self,
                                        function=self._noise_current))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = str(model).upper()

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = node

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = node

    @property
    def node3(self):
        return self.nodes[2]

    @node3.setter
    def node3(self, node):
        self.nodes[2] = node

    def gain(self, frequency):
        """Get op-amp voltage gain at the specified frequency.

        Parameters
        ----------
        frequency : :class:`float`
            Frequency to compute gain at.

        Returns
        -------
        :class:`float`
            Op-amp gain at specified frequency.
        """

        return (self.params["a0"]
                / (1 + self.params["a0"] * 1j * frequency / self.params["gbw"])
                * np.exp(-1j * 2 * np.pi * self.params["delay"] * frequency)
                * np.prod(1 + 1j * frequency / self.params["zeros"])
                / np.prod(1 + 1j * frequency / self.params["poles"]))

    def inverse_gain(self, *args, **kwargs):
        return 1 / self.gain(*args, **kwargs)

    def _noise_voltage(self, component, frequencies):
        return self.params["vn"] * np.sqrt(1 + self.params["vc"] / frequencies)

    def _noise_current(self, node, frequencies):
        # ignore node; noise is same at both inputs
        return self.params["in"] * np.sqrt(1 + self.params["ic"] / frequencies)

    @property
    def voltage_noise(self):
        for noise in self.noise:
            if noise.SUBTYPE == "voltage":
                return noise
        
        raise ValueError("no voltage noise")

    @property
    def non_inv_current_noise(self):
        for noise in self.noise:
            if noise.SUBTYPE == "current":
                if noise.node == self.node1:
                    return noise
        
        raise ValueError("no non-inverting current noise")

    @property
    def inv_current_noise(self):
        for noise in self.noise:
            if noise.SUBTYPE == "current":
                if noise.node == self.node2:
                    return noise
        
        raise ValueError("no inverting current noise")

    def __str__(self):
        return super().__str__() + " [in+={cmp.node1}, in-={cmp.node2}, out={cmp.node3}, model={cmp.model}]".format(cmp=self)

class Input(Component):
    """Represents the circuit's voltage input"""

    TYPE = "input"

    def __init__(self, input_type, node=None, node_p=None, node_n=None,
                 impedance=None, *args, **kwargs):
        # handle nodes
        if node is not None:
            if node_p is not None or node_n is not None:
                raise ValueError("node cannot be specified alongside node_p or "
                                 "node_n")

            nodes = [Node("gnd"), node]
        else:
            if node_p is None or node_n is None:
                raise ValueError("node_p and node_n must both be specified")

            nodes = [node_n, node_p]

        # call parent constructor
        super().__init__(name="input", nodes=nodes, *args, **kwargs)

        input_type = input_type.lower()

        if input_type == "noise":
            self.input_type = "noise"

            if impedance is None:
                raise ValueError("impedance must be specified for noise input")
            
            impedance, _ = SIFormatter.parse(impedance)
        else:
            if impedance is not None:
                raise ValueError("impedance cannot be specified for non-noise input")

            if input_type == "voltage":
                self.input_type = "voltage"
            elif input_type == "current":
                self.input_type = "current"
                # assume 1 ohm impedance for transfer functions
                impedance = 1
            else:
                raise ValueError("unrecognised input type")

        self.impedance = impedance

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = node

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = node

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
        
        return super().__str__() + " [in={cmp.node1}, out={cmp.node2}, Z={z}]".format(cmp=self, z=z)

class Resistor(PassiveComponent):
    """Represents a resistor or set of series or parallel resistors"""

    UNIT = "Ω"
    TYPE = "resistor"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # register Johnson noise
        self.add_noise(JohnsonNoise(component=self, resistance=self.resistance))

    @property
    def resistance(self):
        """Resistance in ohms."""
        return self.value

    @resistance.setter
    def resistance(self, resistance):
        self.value = float(resistance)

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
            if noise.SUBTYPE == "johnson":
                return noise
        
        raise ValueError("no Johnson noise")

    def __str__(self):
        return super().__str__() + " [in={cmp.node1}, out={cmp.node2}, R={cmp.resistance}]".format(cmp=self)

class Capacitor(PassiveComponent):
    """Represents a capacitor or set of series or parallel capacitors"""

    UNIT = "F"
    TYPE = "capacitor"

    @property
    def capacitance(self):
        """Capacitance in farads."""
        return self.value

    @capacitance.setter
    def capacitance(self, capacitance):
        self.value = float(capacitance)

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
        return 1 / (2 * np.pi * 1j * frequency * self.capacitance)

    def __str__(self):
        return super().__str__() + " [in={cmp.node1}, out={cmp.node2}, C={cmp.capacitance}]".format(cmp=self)

class Inductor(PassiveComponent):
    """Represents an inductor or set of series or parallel inductors"""

    UNIT = "H"
    TYPE = "inductor"

    @property
    def inductance(self):
        """Inductance in henries."""
        return self.value

    @inductance.setter
    def inductance(self, inductance):
        self.value = float(inductance)

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
        return 2 * np.pi * 1j * frequency * self.inductance

    def __str__(self):
        return super().__str__() + " [in={cmp.node1}, out={cmp.node2}, L={cmp.inductance}]".format(cmp=self)

class Node(object, metaclass=NamedInstance):
    """Represents a circuit node (connection between components)

    Nodes are considered equal if they have the same case-independent name.

    Parameters
    ----------
    name : :class:`str`
        Node name.

    Attributes
    ----------
    sources : :class:`set`
        :class:`Components <Component>` that source current connected to this
        node.
    sinks : :class:`set`
        :class:`Components <Component>` that sink current connected to this
        node.
    """

    def __init__(self, name):
        """Instantiate a new node."""

        self.name = str(name)

    def label(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class Noise(object, metaclass=abc.ABCMeta):
    """Noise spectral density.

    Parameters
    ----------
    function : callable
        Callable that returns the noise associated with a specified frequency
        vector.
    """

    TYPE = ""

    def __init__(self, function=None):
        self.function = function

    @abc.abstractmethod
    def spectral_density(self, frequencies):
        return NotImplemented

    @abc.abstractmethod
    def label(self):
        return NotImplemented

    def __str__(self):
        return self.label()

    def __repr__(self):
        return str(self)

class ComponentNoise(Noise, metaclass=abc.ABCMeta):
    """Component noise spectral density.

    Parameters
    ----------
    component : :class:`Component`
        Component associated with the noise.
    """

    TYPE = "component"

    def __init__(self, component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.component = component

    def spectral_density(self, frequencies):
        return self.function(component=self.component, frequencies=frequencies)

class NodeNoise(Noise, metaclass=abc.ABCMeta):
    """Node noise spectral density.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the noise.
    component : :class:`Component`
        Component associated with the noise.
    """

    TYPE = "node"

    def __init__(self, node, component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = node
        self.component = component

    def spectral_density(self, *args, **kwargs):
        return self.function(node=self.node, *args, **kwargs)

class VoltageNoise(ComponentNoise):
    SUBTYPE = "voltage"

    def label(self):
        return "V(%s)" % self.component.name

class JohnsonNoise(VoltageNoise):
    SUBTYPE = "johnson"

    def __init__(self, resistance, *args, **kwargs):
        super().__init__(function=self.noise_voltage, *args, **kwargs)

        self.resistance = float(resistance)

    def noise_voltage(self, frequencies, *args, **kwargs):
        white_noise = np.sqrt(4 * float(CONF["constants"]["kB"])
                                * float(CONF["constants"]["T"])
                                * self.resistance)

        return np.ones(frequencies.shape) * white_noise

    def label(self):
        return "R(%s)" % self.component.name

class CurrentNoise(NodeNoise):
    SUBTYPE = "current"

    def label(self):
        return "I(%s, %s)" % (self.component.name, self.node.name)