"""Electronic components"""

import abc
import logging
import numpy as np

from ..misc import Singleton, NamedInstance, _print_progress
from ..format import SIFormatter
from ..config import ElectronicsConfig

CONF = ElectronicsConfig()

class Component(object, metaclass=abc.ABCMeta):
    """Class representing a circuit component"""

    def __init__(self, name=None, nodes=None):
        """Instantiate a new component

        :param name: component name
        :type name: str
        :param nodes: associated component nodes or node names
        :type nodes: Sequence[:class:`~Node`] or Sequence[str]
        """

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
        self.noise.add(noise)

    @abc.abstractmethod
    def label(self):
        return NotImplemented

class PassiveComponent(Component, metaclass=abc.ABCMeta):
    """Represents a passive component"""

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

    def label(self):
        """Label for this passive component"""
        return SIFormatter.format(self.value, self.UNIT)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.label()

    @abc.abstractmethod
    def impedance(self, frequency):
        return NotImplemented

class OpAmp(Component):
    """Represents an (almost) ideal op-amp"""

    def __init__(self, model, node1, node2, node3, a0=1e12, gbw=1e15,
                 delay=1e-9, zeros=np.array([]), poles=np.array([]),
                 v_noise=0, i_noise=0, v_corner=1, i_corner=1, v_max=12,
                 i_max=0.02, slew_rate=1e12, *args, **kwargs):
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

    def label(self):
        return self.name

class Input(PassiveComponent, metaclass=Singleton):
    """Represents the circuit input"""

    def __init__(self, *args, **kwargs):
        super(Input, self).__init__(name="input", *args, **kwargs)

    def impedance(self, *args):
        return 0

    def label(self):
        return "input"

class Resistor(PassiveComponent):
    """Represents a resistor or set of series or parallel resistors"""

    UNIT = "Ω"

    def __init__(self, *args, **kwargs):
        super(Resistor, self).__init__(*args, **kwargs)

        # register Johnson noise
        self.add_noise(JohnsonNoise(component=self, resistance=self.resistance))

    @property
    def resistance(self):
        """Get resistance in ohms"""

        return self.value

    @resistance.setter
    def resistance(self, resistance):
        """Set resistance

        :param resistance: resistance, in ohms
        """

        self.value = float(resistance)

    def impedance(self, *args):
        return self.resistance

class Capacitor(PassiveComponent):
    """Represents a capacitor or set of series or parallel capacitors"""

    UNIT = "F"

    @property
    def capacitance(self):
        """Get capacitance in farads"""

        return self.value

    @capacitance.setter
    def capacitance(self, capacitance):
        """Set capacitance

        :param capacitance: capacitance, in farads
        """

        self.value = float(capacitance)

    def impedance(self, frequency):
        return 1 / (2 * np.pi * 1j * frequency * self.capacitance)

class Inductor(PassiveComponent):
    """Represents an inductor or set of series or parallel inductors"""

    UNIT = "H"

    @property
    def inductance(self):
        """Get inductance in henries"""

        return self.value

    @inductance.setter
    def inductance(self, inductance):
        """Set inductance

        :param inductance: inductance, in henries
        """

        self.value = float(inductance)

    def impedance(self, frequency):
        return 2 * np.pi * 1j * frequency * self.inductance

class Node(object, metaclass=NamedInstance):
    """Represents a circuit node (connection between components)

    Nodes are considered equal if they have the same case-independent name.
    """

    def __init__(self, name):
        """Instantiate a new node

        :param name: node name
        """

        self.name = str(name)

        # current sources and sinks
        self.sources = set()
        self.sinks = set()

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

    def reset(self):
        """Remove all references to external objects

        This is useful for when a node is re-used in a different circuit by
        the same Python kernel. The named instance pattern results in the same
        object as before being returned by the constructor, with references to
        old circuits. Removing these references avoids bad references.
        """

        self.sources.clear()
        self.sinks.clear()

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class Noise(object, metaclass=abc.ABCMeta):
    """Noise spectral density"""

    def __init__(self, function=None):
        self.function = function

    @abc.abstractmethod
    def spectral_density(self, frequencies):
        return NotImplemented

    @abc.abstractmethod
    def label(self):
        return NotImplemented

class ComponentNoise(Noise, metaclass=abc.ABCMeta):
    """Component noise spectral density"""

    def __init__(self, component, *args, **kwargs):
        super(ComponentNoise, self).__init__(*args, **kwargs)

        self.component = component

    def spectral_density(self, frequencies):
        return self.function(component=self.component, frequencies=frequencies)

    def __str__(self):
        return "%s[%s]" % (self.label(), self.component.name)

class NodeNoise(Noise, metaclass=abc.ABCMeta):
    """Node noise spectral density"""

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
    def __init__(self, coefficients):
        """Instantiate a new equation

        :param coefficients: coefficients that make up the equation
        """

        self.coefficients = []

        for coefficient in coefficients:
            self.add_coefficient(coefficient)

    def add_coefficient(self, coefficient):
        """Add coefficient to equation

        :param coefficient: coefficient to add
        """

        self.coefficients.append(coefficient)

class ComponentEquation(BaseEquation):
    def __init__(self, component, *args, **kwargs):
        """Instantiate a new component equation

        :param component: component this equation represents
        """

        # call parent constructor
        super(ComponentEquation, self).__init__(*args, **kwargs)

        self.component = component

class NodeEquation(BaseEquation):
    def __init__(self, node, *args, **kwargs):
        """Instantiate a new node equation

        :param node: node this equation represents
        """

        # call parent constructor
        super(NodeEquation, self).__init__(*args, **kwargs)

        self.node = node

class BaseCoefficient(object, metaclass=abc.ABCMeta):
    # coefficient types
    TYPE_IMPEDANCE = 0
    TYPE_CURRENT = 1
    TYPE_VOLTAGE = 2

    def __init__(self, value, coefficient_type):
        """Instantiate a new coefficient

        :param value: coefficient value, which may be either a number or a \
                      callable which returns a number
        :param coefficient_type: coefficient type number
        """

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
    def __init__(self, component, *args, **kwargs):
        """Instantiate a new impedance coefficient

        :param component: component this impedance corresponds to
        """

        self.component = component

        # call parent constructor
        return super(ImpedanceCoefficient, self).__init__(
            coefficient_type=self.TYPE_IMPEDANCE, *args, **kwargs)

class CurrentCoefficient(BaseCoefficient):
    def __init__(self, component, *args, **kwargs):
        """Instantiate a new current coefficient

        :param component: component this current corresponds to
        """

        self.component = component

        # call parent constructor
        return super(CurrentCoefficient, self).__init__(
            coefficient_type=self.TYPE_CURRENT, *args, **kwargs)

class VoltageCoefficient(BaseCoefficient):
    def __init__(self, node, *args, **kwargs):
        """Instantiate a new voltage coefficient

        :param node: node this voltage corresponds to
        """

        self.node = node

        # call parent constructor
        return super(VoltageCoefficient, self).__init__(
            coefficient_type=self.TYPE_VOLTAGE, *args, **kwargs)
