"""Electronic components"""

import abc
import logging
import numpy as np

from .misc import Singleton, NamedInstance, _print_progress
from .format import SIFormatter
from .config import CircuitConfig

CONF = CircuitConfig()

class Component(object, metaclass=abc.ABCMeta):
    """Represents a circuit component.

    Parameters
    ----------
    name : :class:`str`
        component name
    nodes : sequence of :class:`~Node` or :class:`str`
        component nodes

    Attributes
    ----------
    noise : :class:`set`
        component noise sources
    """

    def __init__(self, name=None, nodes=None):
        """Instantiate a new component."""

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
        """Component nodes.

        Returns
        -------
        list of :class:`~Node`
            list of component nodes
        """

        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        for node in list(nodes):
            if not isinstance(node, Node):
                # parse node name
                node = Node(str(node))

            self._nodes.append(node)

    @abc.abstractmethod
    def equation(self):
        return NotImplemented

    def add_noise(self, noise):
        """Add a noise source to the component.

        Parameters
        ----------
        noise : :class:`~Noise`
            noise to add
        """

        self.noise.add(noise)

    def label(self):
        """Label for this passive component.

        Returns
        -------
        :class:`str`
            Component label
        """

        return self.name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.label()

class PassiveComponent(Component, metaclass=abc.ABCMeta):
    """Represents a passive component.

    A passive component is one that consumes or temporarily stores energy, but
    does not produce or amplify it. Examples include
    :class:`resistors <Resistor>`, :class:`capacitors <Capacitor>` and
    :class:`inductors <Inductor>`.

    Parameters
    ----------
    value : any
        Component value.
    node1 : :class:`~Node`
        First component node.
    node2 : :class:`~Node`
        Second component node.

    Attributes
    ----------
    value : :class:`float`
        Component value.
    """

    UNIT = "?"

    def __init__(self, value=None, node1=None, node2=None, *args, **kwargs):
        super(PassiveComponent, self).__init__(nodes=[node1, node2], *args,
                                               **kwargs)

        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            value, _ = SIFormatter.parse(value)

        self._value = value

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        # FIXME: remove old node's sources/sinks
        self.nodes[0] = node

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        # FIXME: remove old node's sources/sinks
        self.nodes[1] = node

    def equation(self):
        """Get component equation.

        The component equation represents the component in the
        :class:`circuit <.circuit.Circuit>` matrix. It maps input or noise
        sources to voltages across :class:`nodes <Node>` and currents through
        :class:`components <Component>` in terms of their
        :class:`impedance <ImpedanceCoefficient>` and
        :class:`voltage <VoltageCoefficient>` coefficients.

        This method also registers current sources and sinks with the
        component's :class:`nodes <Node>`.

        Returns
        -------
        :class:`~ComponentEquation`
            Component equation containing :class:`coefficients <BaseCoefficient>`.
        """

        # register component as sink for node 1 and source for node 2
        if self.node1:
            self.node1.add_sink(self) # current flows into here...
        if self.node2:
            self.node2.add_source(self) # and out of here

        # nodal potential equation coefficients
        # impedance * current + voltage = 0
        coefficients = []

        # add impedance
        coefficients.append(ImpedanceCoefficient(component=self,
                                                 value=self.impedance))

        # add input node coefficient
        if self.node1 is not Node("gnd"):
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node1,
                                                   value=-1))

        # add output node coefficient
        if self.node2 is not Node("gnd"):
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node2,
                                                   value=1))

        # create and return equation
        return ComponentEquation(self, coefficients=coefficients)

    @abc.abstractmethod
    def impedance(self, frequency):
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
    a0 : :class:`float`
        Open loop gain.
    gbw : :class:`float`
        Gain-bandwidth product.
    delay : :class:`float`
        Delay.
    zeros : :class:`np.ndarray`
        Zeros.
    poles : :class:`np.ndarray`
        Poles.
    v_noise : :class:`float`
        Flat voltage noise.
    i_noise : :class:`float`
        Float current noise.
    v_corner : :class:`float`
        Voltage noise corner frequency.
    i_corner : :class:`float`
        Current noise corner frequency.
    v_max : :class:`float`
        Maximum input voltage.
    i_max : :class:`float`
        Maximum output current.
    slew_rate : :class:`float`
        Slew rate.
    """

    def __init__(self, model, node1, node2, node3, a0=1.5e6, gbw=8e6,
                 delay=0, zeros=np.array([]), poles=np.array([]),
                 v_noise=3.2e-9, i_noise=0.4e-12, v_corner=2.7, i_corner=140,
                 v_max=12, i_max=0.06, slew_rate=1e6, *args, **kwargs):
        """"Instantiate new op-amp."""

        # call parent constructor
        super(OpAmp, self).__init__(nodes=[node1, node2, node3], *args, **kwargs)

        # default properties
        self.params = {"a0": SIFormatter.parse(a0)[0], # gain
                       "gbw": SIFormatter.parse(gbw)[0], # gain-bandwidth product (Hz)
                       "delay": SIFormatter.parse(delay)[0], # delay (s)
                       "zeros": zeros, # array of additional zeros
                       "poles": poles, # array of additional poles
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
            self.add_noise(CurrentNoise(node=self.node1, component=self,
                                        function=self._noise_current))
        if self.node2 is not Node("gnd"):
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

    def equation(self):
        """Get op-amp equation."""

        # register component as source for node 3
        # nodes 1 and 2 don't source or sink current (ideally)
        self.node3.add_source(self) # current flows out of here

        # nodal potential equation coefficients
        # V[n3] = H(s) (V[n1] - V[n2])
        coefficients = []

        # add non-inverting input node coefficient
        if self.node1 is not Node("gnd"):
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node1,
                                                   value=-1))

        # add inverting input node coefficient
        if self.node2 is not Node("gnd"):
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node2,
                                                   value=1))

        # add output node coefficient
        if self.node3 is not Node("gnd"):
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node3,
                                                   value=self.inverse_gain))

        # create and return equation
        return ComponentEquation(self, coefficients=coefficients)

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

class Input(Component):
    """Represents the circuit's voltage input"""

    TYPE_NOISE = 1
    TYPE_VOLTAGE = 2
    TYPE_CURRENT = 3

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
        super(Input, self).__init__(name="input", nodes=nodes, *args,
                                    **kwargs)

        # default value
        self.defaults()

        if input_type in ["noise", self.TYPE_NOISE]:
            self.input_type = self.TYPE_NOISE
            if impedance is None:
                raise ValueError("impedance must be specified for noise input")
            self.impedance = float(impedance)
        else:
            if impedance is not None:
                raise ValueError("impedance cannot be specified for non-noise "
                                 "input")

            if input_type in ["voltage", self.TYPE_VOLTAGE]:
                self.input_type = self.TYPE_VOLTAGE
            elif input_type in ["current", self.TYPE_CURRENT]:
                self.input_type = self.TYPE_CURRENT
                # assume 1 ohm impedance for transfer functions
                self.impedance = 1
            else:
                raise ValueError("unrecognised input type")

    @property
    def node_n(self):
        return self.nodes[0]

    @node_n.setter
    def node_n(self, node):
        self.nodes[0] = node

    @property
    def node_p(self):
        return self.nodes[1]

    @node_p.setter
    def node_p(self, node):
        self.nodes[1] = node

    def equation(self):
        # register component as sink for negative node and source for positive
        # node
        if self.node_n:
            self.node_n.add_sink(self) # current flows into here...
        if self.node_p:
            self.node_p.add_source(self) # and out of here

        # nodal potential equation coefficients
        # impedance * current + voltage = 0
        coefficients = []

        if self.input_type in [self.TYPE_NOISE, self.TYPE_CURRENT]:
            # set impedance
            coefficients.append(ImpedanceCoefficient(component=self,
                                                     value=self.impedance))

        if self.input_type in [self.TYPE_NOISE, self.TYPE_VOLTAGE]:
            # add input node coefficient
            if self.node_n is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=self.node_n,
                                                       value=-1))

            # add output node coefficient
            if self.node_p is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=self.node_p,
                                                       value=1))

        # create and return equation
        return ComponentEquation(self, coefficients=coefficients)

    def defaults(self):
        """Restore default settings

        This is useful for when a node is re-used in a different circuit by
        the same Python kernel. The singleton pattern results in the same
        object as before being returned by the constructor, with references to
        old circuits. Removing these references avoids bad references.
        """

        self.impedance = None

class Resistor(PassiveComponent):
    """Represents a resistor or set of series or parallel resistors"""

    UNIT = "Ω"

    def __init__(self, *args, **kwargs):
        super(Resistor, self).__init__(*args, **kwargs)

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
        return self.resistance

class Capacitor(PassiveComponent):
    """Represents a capacitor or set of series or parallel capacitors"""

    UNIT = "F"

    @property
    def capacitance(self):
        """Capacitance in farads."""
        return self.value

    @capacitance.setter
    def capacitance(self, capacitance):
        self.value = float(capacitance)

    def impedance(self, frequency):
        return 1 / (2 * np.pi * 1j * frequency * self.capacitance)

class Inductor(PassiveComponent):
    """Represents an inductor or set of series or parallel inductors"""

    UNIT = "H"

    @property
    def inductance(self):
        """Inductance in henries."""
        return self.value

    @inductance.setter
    def inductance(self, inductance):
        self.value = float(inductance)

    def impedance(self, frequency):
        return 2 * np.pi * 1j * frequency * self.inductance

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

        # default settings
        self.defaults()

    def add_source(self, component):
        self.sources.add(component)

    def add_sink(self, component):
        self.sinks.add(component)

    def equation(self):
        # nodal current equation coefficients
        # current out - current in = 0
        coefficients = []

        for source in self.sources:
            # add source coefficient
            if source is not Node("gnd"):
                coefficients.append(CurrentCoefficient(component=source,
                                                       value=1))

        for sink in self.sinks:
            # add sink coefficient
            if sink is not Node("gnd"):
                coefficients.append(CurrentCoefficient(component=sink,
                                                       value=-1))

        # create and return equation
        return NodeEquation(self, coefficients=coefficients)

    def defaults(self):
        """Restore default settings.

        This is useful for when a node is re-used in a different circuit by
        the same Python kernel. The named instance pattern results in the same
        object as before being returned by the constructor, with references to
        old circuits. Removing these references avoids bad references.
        """

        self.sources = set()
        self.sinks = set()

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

    def __init__(self, function=None):
        self.function = function

    @abc.abstractmethod
    def spectral_density(self, frequencies):
        return NotImplemented

    @abc.abstractmethod
    def label(self):
        return NotImplemented

class ComponentNoise(Noise, metaclass=abc.ABCMeta):
    """Component noise spectral density.

    Parameters
    ----------
    component : :class:`Component`
        Component associated with the noise.
    """

    def __init__(self, component, *args, **kwargs):
        super(ComponentNoise, self).__init__(*args, **kwargs)

        self.component = component

    def spectral_density(self, frequencies):
        return self.function(component=self.component, frequencies=frequencies)

    def __str__(self):
        return "%s[%s]" % (self.label(), self.component.name)

class NodeNoise(Noise, metaclass=abc.ABCMeta):
    """Node noise spectral density.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the noise.
    component : :class:`Component`
        Component associated with the noise.
    """

    def __init__(self, node, component, *args, **kwargs):
        super(NodeNoise, self).__init__(*args, **kwargs)

        self.node = node
        self.component = component

    def spectral_density(self, *args, **kwargs):
        return self.function(node=self.node, *args, **kwargs)

    def __str__(self):
        return "%s[%s, %s]" % (self.label(), self.component.name, self.node.name)

class VoltageNoise(ComponentNoise):
    def label(self):
        return "VNoise"

class JohnsonNoise(VoltageNoise):
    def __init__(self, resistance, *args, **kwargs):
        super(JohnsonNoise, self).__init__(function=self.noise_voltage, *args,
                                           **kwargs)

        self.resistance = float(resistance)

    def noise_voltage(self, frequencies, *args, **kwargs):
        white_noise = np.sqrt(4 * float(CONF["constants"]["kB"])
                                * float(CONF["constants"]["T"])
                                * self.resistance)

        return np.ones(frequencies.shape) * white_noise

    def label(self):
        return "RNoise"

class CurrentNoise(NodeNoise):
    def label(self):
        return "INoise"

class BaseEquation(object, metaclass=abc.ABCMeta):
    """Represents an equation.

    Parameters
    ----------
    coefficients : sequence of :class:`BaseCoefficient`
        Coefficients that make up the equation.
    """

    def __init__(self, coefficients):
        """Instantiate a new equation."""

        self.coefficients = []

        for coefficient in coefficients:
            self.add_coefficient(coefficient)

    def add_coefficient(self, coefficient):
        """Add coefficient to equation.

        Parameters
        ----------
        coefficient : :class:`BaseCoefficient`
            Coefficient to add.
        """

        self.coefficients.append(coefficient)

class ComponentEquation(BaseEquation):
    """Represents a component equation.

    Parameters
    ----------
    component : :class:`Component`
        Component associated with the equation.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new component equation."""

        # call parent constructor
        super(ComponentEquation, self).__init__(*args, **kwargs)

        self.component = component

class NodeEquation(BaseEquation):
    """Represents a node equation.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the equation.
    """

    def __init__(self, node, *args, **kwargs):
        """Instantiate a new node equation."""

        # call parent constructor
        super(NodeEquation, self).__init__(*args, **kwargs)

        self.node = node

class BaseCoefficient(object, metaclass=abc.ABCMeta):
    """Represents a coefficient.

    Parameters
    ----------
    value : :class:`float`
        Coefficient value.
    coefficient_type : {0, 1, 2}
        Coefficient type. Impedance is 0, current is 1, voltage is 2.
    """

    # coefficient types
    TYPE_IMPEDANCE = 0
    TYPE_CURRENT = 1
    TYPE_VOLTAGE = 2

    def __init__(self, value, coefficient_type):
        """Instantiate a new coefficient."""

        self.value = value
        self.coefficient_type = coefficient_type

    @property
    def coefficient_type(self):
        return self._coefficient_type

    @coefficient_type.setter
    def coefficient_type(self, coefficient_type):
        # validate coefficient
        if coefficient_type not in [self.TYPE_IMPEDANCE,
                                    self.TYPE_CURRENT,
                                    self.TYPE_VOLTAGE]:
            raise ValueError("Unrecognised coefficient type")

        self._coefficient_type = coefficient_type

class ImpedanceCoefficient(BaseCoefficient):
    """Represents an impedance coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this impedance coefficient represents.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new impedance coefficient."""

        self.component = component

        # call parent constructor
        super(ImpedanceCoefficient, self).__init__(
              coefficient_type=self.TYPE_IMPEDANCE, *args, **kwargs)

class CurrentCoefficient(BaseCoefficient):
    """Represents an current coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this current coefficient represents.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new current coefficient."""

        self.component = component

        # call parent constructor
        super(CurrentCoefficient, self).__init__(
              coefficient_type=self.TYPE_CURRENT, *args, **kwargs)

class VoltageCoefficient(BaseCoefficient):
    """Represents a voltage coefficient.

    Parameters
    ----------
    node : :class:`Node`
        Node this voltage coefficient represents.
    """

    def __init__(self, node, *args, **kwargs):
        """Instantiate a new voltage coefficient."""

        self.node = node

        # call parent constructor
        super(VoltageCoefficient, self).__init__(
              coefficient_type=self.TYPE_VOLTAGE, *args, **kwargs)
