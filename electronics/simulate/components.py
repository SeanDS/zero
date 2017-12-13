"""Electronic components"""

import abc
import logging
import numpy as np
import itertools
import heapq

from ..misc import Singleton, _n_comb_k, _print_progress
from ..format import SIFormatter
from ..config import ElectronicsConfig, OpAmpLibrary

CONF = ElectronicsConfig()
LIBRARY = OpAmpLibrary()

class Component(object, metaclass=abc.ABCMeta):
    """Class representing a circuit component"""

    def __init__(self, name=None, nodes=None, noise_current_nodes=None):
        """Instantiate a new component

        :param name: component name
        :type name: str
        :param nodes: associated component nodes
        :type nodes: Sequence[:class:`~Node`]
        :param noise_current_nodes: nodes which contribute to current noise
        :type noise_current_nodes: Sequence[:class:`~Node`]
        """

        if name is not None:
            name = str(name)

        if nodes is None:
            nodes = []

        self.name = name
        self.nodes = list(nodes)
        self.noise_current_nodes = noise_current_nodes

    def noise_sources(self):
        # no noise sources by default
        return []

    def noise_voltage(self, frequency):
        # no noise by default
        return 0

    def noise_currents(self, frequency):
        return {node: self.node_noise_current(node, frequency)
                for node in self.noise_current_nodes}

    def node_noise_current(self, node, frequency):
        return 0

    @property
    def noise_current_nodes(self):
        return self._noise_current_nodes

    @noise_current_nodes.setter
    def noise_current_nodes(self, nodes):
        if nodes is None:
            nodes = []

        nodes = list(nodes)

        if not set(nodes).issubset(set(self.nodes)):
            raise ValueError("A noise current node was specified that is"
                             "not a node of this component")

        self._noise_current_nodes = nodes

    @abc.abstractmethod
    def equation(self):
        return NotImplemented

    @abc.abstractmethod
    def label(self):
        return NotImplemented

class PassiveComponent(Component, metaclass=abc.ABCMeta):
    """Represents a passive component"""

    UNIT = "?"

    def __init__(self, value=None, tolerance=None, node1=None, node2=None,
                 *args, **kwargs):
        super(PassiveComponent, self).__init__(nodes=[node1, node2], *args,
                                               **kwargs)

        self.value = value
        self.tolerance = tolerance

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            value = SIFormatter.parse(value)

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
        if self.node1 is not Gnd():
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node1,
                                                   value=-1))

        # add output node coefficient
        if self.node2 is not Gnd():
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node2,
                                                   value=1))

        # create and return equation
        return ComponentEquation(self, coefficients=coefficients)

    @property
    def tolerance(self):
        """Get tolerance in percent"""

        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        """Set tolerance

        :param tolerance: tolerance, in percent
        """

        if tolerance is not None:
            tolerance = float(tolerance)

        self._tolerance = tolerance

    @property
    def abs_tolerance(self):
        """Absolute tolerance"""
        return self.value * self.tolerance / 100

    @property
    def abs_inv_tolerance(self):
        """Absolute inverse tolerance"""
        return (1 / self.value) * self.tolerance / 100

    def label(self):
        """Label for this passive component"""

        label = SIFormatter.format(self.value, self.UNIT)

        if self.tolerance:
            label += " ± {}%".format(SIFormatter.format(self.tolerance))

        return label

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.label()

    @abc.abstractmethod
    def impedance(self, frequency):
        return NotImplemented

class OpAmp(Component):
    """Represents an (almost) ideal op-amp"""

    def __init__(self, model=None, node1=None, node2=None, node3=None, *args,
                 **kwargs):
        # call parent constructor
        super(OpAmp, self).__init__(nodes=[node1, node2, node3], *args, **kwargs)

        # default properties
        # DC gain
        self._a0 = 1e12
        # gain-bandwidth product (Hz)
        self._gbw = 1e15
        # delay (s)
        self._delay = 1e-9
        # array of additional zeros (Hz)
        self._zeros = np.array([])
        # array of additional poles
        self._poles = np.array([])
        # voltage noise (V/sqrt(Hz))
        self._vn = 0
        # current noise (A/sqrt(Hz))
        self._in = 0
        # voltage noise corner frequency (Hz)
        self._vc = 1
        # current noise corner frequency (Hz)
        self._iv = 1
        # maximum output voltage amplitude (V)
        self._vmax = 12
        # maximum output current amplitude (A)
        self._imax = 0.02
        # maximum slew rate (V/s)
        self._sr = 1e12

        # set model and populate properties
        self.model = model

        # update noise current nodes in case they've been set
        self._update_noise_current_nodes()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        model = str(model).upper()

        if not LIBRARY.has_data(model):
            raise ValueError("Unrecognised op-amp type: %s" % model)

        self._model = model

        # set data
        self._set_model_data(model)

    def _set_model_data(self, model):
        # get library data
        data = LIBRARY.get_data(model)

        if "a0" in data:
            self._a0 = data["a0"]
        if "gbw" in data:
            self._gbw = data["gbw"]
        if "delay" in data:
            self._delay = data["delay"]
        if "zeros" in data:
            self._zeros = data["zeros"]
        if "poles" in data:
            self._poles = data["poles"]
        if "vn" in data:
            self._vn = data["vn"]
        if "in" in data:
            self._in = data["in"]
        if "vc" in data:
            self._vc = data["vc"]
        if "ic" in data:
            self._ic = data["ic"]
        if "vmax" in data:
            self._vmax = data["vmax"]
        if "imax" in data:
            self._imax = data["imax"]
        if "sr" in data:
            self._sr = data["sr"]

    @property
    def node1(self):
        return self.nodes[0]

    @node1.setter
    def node1(self, node):
        self.nodes[0] = node
        self._update_noise_current_nodes()

    @property
    def node2(self):
        return self.nodes[1]

    @node2.setter
    def node2(self, node):
        self.nodes[1] = node
        self._update_noise_current_nodes()

    @property
    def node3(self):
        return self.nodes[2]

    @node3.setter
    def node3(self, node):
        self.nodes[2] = node
        self._update_noise_current_nodes()

    def _update_noise_current_nodes(self):
        noise_nodes = []

        # input nodes contribute to noise unless they're grounded
        if self.node1 is not None and self.node1 != Gnd():
            noise_nodes.append(self.node1)
        if self.node2 is not None and self.node2 != Gnd():
            noise_nodes.append(self.node2)

        self.noise_current_nodes = noise_nodes

    def equation(self):
        # register component as source for node 3
        # nodes 1 and 2 don't source or sink current (ideally)
        self.node3.add_source(self) # current flows out of here

        # nodal potential equation coefficients
        # V[n3] = H(s) (V[n1] - V[n2])
        coefficients = []

        # add non-inverting input node coefficient
        if self.node1 is not Gnd():
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node1,
                                                   value=-1))

        # add inverting input node coefficient
        if self.node2 is not Gnd():
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node2,
                                                   value=1))

        # add output node coefficient
        if self.node3 is not Gnd():
            # voltage
            coefficients.append(VoltageCoefficient(node=self.node3,
                                                   value=self.inverse_gain))

        # create and return equation
        return ComponentEquation(self, coefficients=coefficients)

    def gain(self, frequency):
        return (self._a0 / (1 + self._a0 * 1j * frequency / self._gbw)
                * np.exp(-1j * 2 * np.pi * self._delay * frequency)
                * np.prod(1 + 1j * frequency / self._zeros)
                / np.prod(1 + 1j * frequency / self._poles))

    def inverse_gain(self, *args, **kwargs):
        return 1 / self.gain(*args, **kwargs)

    def noise_voltage(self, frequency):
        return self._vn * np.sqrt(1 + self._vc / frequency)

    def node_noise_current(self, node, frequency):
        # ignore node; noise is same at both
        return self._in * np.sqrt(1 + self._ic / frequency)

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

    # tolerances, in percent (+/-)
    TOL_GREY = 0.05
    TOL_VIOLET = 0.1
    TOL_BLUE = 0.25
    TOL_GREEN = 0.5
    TOL_BROWN = 1.0
    TOL_RED = 2.0
    TOL_GOLD = 5.0
    TOL_SILVER = 10.0
    TOL_NONE = 20.0

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

    def noise_voltage(self, *args):
        return np.sqrt(4 * float(CONF["constants"]["kB"])
                       * float(CONF["constants"]["T"]) * self.resistance)

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

class ResistorCollection(Resistor):
    """Represents a collection of resistors that behave like a single resistor"""

    # configuration types
    TYPE_SERIES = 1
    TYPE_PARALLEL = 2

    def __init__(self, resistors, vtype=None, *args, **kwargs):
        """Instantiate a resistor collection

        :param resistors: resistors in the collection
        :param vtype: configuration type (series or parallel)
        """

        # call parent constructor
        super(ResistorCollection, self).__init__(*args, **kwargs)

        # default to series
        if vtype is None:
            vtype = self.TYPE_SERIES
        elif vtype not in (self.TYPE_SERIES, self.TYPE_PARALLEL):
            raise ValueError("Unrecognised vtype")

        self.resistors = list(resistors)
        self.vtype = vtype

    @Resistor.value.getter
    def value(self):
        """Calculate resistance in ohms"""

        if self.vtype is self.TYPE_SERIES:
            # series sum
            return sum(resistor.resistance for resistor in self.resistors)
        else:
            # parallel sum
            return 1 / sum(1 / resistor.resistance for resistor in self.resistors)

    @Resistor.tolerance.getter
    def tolerance(self):
        """Tolerance of combination in percent

        Important note: this assumes that that resistors are independently
        random variables. Resistors from the same batch or line will not be
        fully independent. Manufacturers may also select better toreranced
        resistors for other lines, resulting in a non-normal distribution, which
        makes this calculation inaccurate.

        Furthermore, the mean and variance of an inverse distribution - which is
        used for the parallel calculation - is not generally the inverse of the
        original mean and variance. A more accurate calculation would involve a
        Taylor expansion, which we don't do here (see
        http://paulorenato.com/index.php/109).
        """

        if self.vtype is self.TYPE_SERIES:
            # quadrature sum of absolute tolerances
            abs_q_sum = sum([resistor.abs_tolerance ** 2
                             for resistor in self.resistors]) ** 0.5

            # express as percentage of total resistance
            return 100 * abs_q_sum / self.resistance
        else:
            # quadrature sum of absolute inverse tolerances
            abs_q_sum = sum([resistor.abs_inv_tolerance ** 2
                             for resistor in self.resistors]) ** 0.5

            # express as percentage of total inverse resistance
            return 100 * abs_q_sum / sum([1 / resistor.resistance
                                          for resistor in self.resistors])

    def noise_voltage(self, frequency):
        # TODO: add noise voltage calculation
        return NotImplemented

    def series_equivalent(self):
        """Return collection with identical resistors in series"""

        return ResistorCollection(self.resistors, self.TYPE_SERIES)

    def parallel_equivalent(self):
        """Return collection with identical resistors in parallel"""

        return ResistorCollection(self.resistors, self.TYPE_PARALLEL)

    def label(self, constituents=True, *args, **kwargs):
        """Label for this collection

        :param constituents: show constituent resistors and tolerances
        """

        label = super(ResistorCollection, self).label(*args, **kwargs)

        if constituents:
            if self.vtype is self.TYPE_SERIES:
                combined_resistances = " + ".join([r.label(*args, **kwargs)
                                                   for r in self.resistors])
            else:
                combined_resistances = " || ".join([r.label(*args, **kwargs)
                                                    for r in self.resistors])

            label += " ({})".format(combined_resistances)

        return label

class Node(object):
    """Represents a circuit node (connection between components)"""

    def __init__(self, name):
        """Instantiate a new node

        :param name: node name
        """

        self.name = name

        # current sources and sinks
        self.sources = set()
        self.sinks = set()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name = str(name)

        # only Gnd() node can be called "gnd"
        if self.__class__.__name__ == "Node" and name.lower() == "gnd":
            raise ValueError("Ground nodes must be created with Gnd()")

        self._name = name

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
            if source is not Gnd():
                coefficients.append(CurrentCoefficient(component=source,
                                                       value=1))

        for sink in self.sinks:
            # add sink coefficient
            if sink is not Gnd():
                coefficients.append(CurrentCoefficient(component=sink,
                                                       value=-1))

        # create and return equation
        return NodeEquation(self, coefficients=coefficients)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

class Gnd(Node, metaclass=Singleton):
    """Node representing ground potential

    Objects of this class are always the same instance.
    """

    def __init__(self):
        """Instantiate a ground node"""

        # call parent constructor
        return super(Gnd, self).__init__(name="Gnd")

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

class Set(object):
    """Set of resistors and associated operations"""

    # resistor series
    SERIES_E3 = 1
    SERIES_E6 = 2
    SERIES_E12 = 3
    SERIES_E24 = 4
    SERIES_E48 = 5
    SERIES_E96 = 6
    SERIES_E192 = 7
    SERIES_CUSTOM = 8

    # base values for E192 (from which E96 and E48 are derived)
    VALUES_E192 = [
        1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11, 1.13, 1.14,
        1.15, 1.17, 1.18, 1.20, 1.21, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.32,
        1.33, 1.35, 1.37, 1.38, 1.40, 1.42, 1.43, 1.45, 1.47, 1.49, 1.50, 1.52,
        1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76,
        1.78, 1.80, 1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03,
        2.05, 2.08, 2.10, 2.13, 2.15, 2.18, 2.21, 2.23, 2.26, 2.29, 2.32, 2.34,
        2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58, 2.61, 2.64, 2.67, 2.71,
        2.74, 2.77, 2.80, 2.84, 2.87, 2.91, 2.94, 2.98, 3.01, 3.05, 3.09, 3.12,
        3.16, 3.20, 3.24, 3.28, 3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61,
        3.65, 3.70, 3.74, 3.79, 3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17,
        4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59, 4.64, 4.70, 4.75, 4.81,
        4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30, 5.36, 5.42, 5.49, 5.56,
        5.62, 5.69, 5.76, 5.83, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26, 6.34, 6.42,
        6.49, 6.57, 6.65, 6.73, 6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41,
        7.50, 7.59, 7.68, 7.77, 7.87, 7.96, 8.06, 8.16, 8.25, 8.35, 8.45, 8.56,
        8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88]

    # base values for E24 (from which E12, E6 and E3 are derived)
    VALUES_E24 = [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
        3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
    ]

    def __init__(self, series=None, tolerance=None, max_exp=6, min_exp=0,
                 max_series=1, min_series=1, max_parallel=1, min_parallel=1):
        """Initialise resistor set

        :param series: resistor series
        :param tolerance: resistor tolerance
        :param max_exp: maximum exponent
        :param min_exp: minimum exponent
        :param max_series: maximum number of series combinations
        :param min_series: minimum number of series combinations
        :param max_parallel: maximum number of parallel combinations
        :param min_parallel: minimum number of parallel combinations
        """

        if series is None:
            logging.getLogger("resistor").info("Using E24 series by default")
            series = self.SERIES_E24
        else:
            str_series = str(series).lower()

            if str_series == "e3":
                series = self.SERIES_E3
            elif str_series == "e6":
                series = self.SERIES_E6
            elif str_series == "e12":
                series = self.SERIES_E12
            elif str_series == "e24":
                series = self.SERIES_E24
            elif str_series == "e48":
                series = self.SERIES_E48
            elif str_series == "e96":
                series = self.SERIES_E96
            elif str_series == "e192":
                series = self.SERIES_E192

        if tolerance is None:
            logging.getLogger("resistor").info("Using gold (5%) tolerance by default")
            tolerance = Resistor.TOL_GOLD
        else:
            # format tolerance
            tolerance = float(str(tolerance).strip().replace("%", ""))

        self.series = int(series)
        self.tolerance = float(tolerance)
        self.max_exp = int(max_exp)
        self.min_exp = int(min_exp)
        self.max_series = int(max_series)
        self.min_series = int(min_series)
        self.max_parallel = int(max_parallel)
        self.min_parallel = int(min_parallel)

    def _base_numbers(self):
        """Get set's numbers between 1 and 10

        :return: power 1 numbers obeying the object's series
        """

        if self.series is self.SERIES_E192:
            return self.VALUES_E192
        elif self.series is self.SERIES_E96:
            return self.VALUES_E192[::2]
        elif self.series is self.SERIES_E48:
            return self.VALUES_E192[::4]
        elif self.series is self.SERIES_E24:
            return self.VALUES_E24
        elif self.series is self.SERIES_E12:
            return self.VALUES_E24[::2]
        elif self.series is self.SERIES_E6:
            return self.VALUES_E24[::4]
        elif self.series is self.SERIES_E3:
            return self.VALUES_E24[::8]
        else:
            raise ValueError("Unrecognised resistor series")

    def _base_resistors(self):
        """Calculate set's base resistors

        :return: resistors with power 1 values obeying the object's series
        """

        if self.min_exp > self.max_exp:
            raise ValueError("max_exp must be >= min_exp")

        # calculate exponents of the base numbers and fill values set
        for exp in range(self.min_exp, self.max_exp + 1):
            for base in self._base_numbers():
                value = base *10 ** exp
                yield Resistor(value=value, tolerance=self.tolerance)

    def combinations(self):
        """Get series/parallel resistor combinations

        :return: generator which yields non-unique resistor values or \
                 series/parallel combinations of the object's series
        """

        if self.min_series < 2 or self.min_parallel < 2:
            # yield single resistors
            yield from self._base_resistors()

        # compute series and parallel combinations
        yield from self._series_combinations()
        yield from self._parallel_combinations()

    def _series_combinations(self):
        """Get series combinations of the specified set of values

        :return: generator which yields series combinations of the object's \
                 series
        """

        # create series collections
        for resistors in self._value_combinations(self._base_resistors(),
                                                  self.min_series,
                                                  self.max_series):
            yield ResistorCollection(resistors=resistors,
                                     vtype=ResistorCollection.TYPE_SERIES)

    def _parallel_combinations(self):
        """Get parallel combinations of the specified set of values

        :return: generator which yields parallel combinations of the object's \
                 series
        """

        # create series collections
        for resistors in self._value_combinations(self._base_resistors(),
                                                  self.min_parallel,
                                                  self.max_parallel):
            yield ResistorCollection(resistors=resistors,
                                     vtype=ResistorCollection.TYPE_PARALLEL)

    @staticmethod
    def _value_combinations(values, min_count, max_count):
        """Returns combinations of the specified set of values between \
        min_count and max_count

        :param values: set of values
        :param min_count: minimum number of values in each combination
        :param max_count: maximum number of values in each combination
        :return: combinations of values
        """

        values = list(values)

        min_count = int(min_count)
        max_count = int(max_count)

        if min_count > max_count:
            raise ValueError("max_count must be >= min_count")
        elif max_count < 2:
            return []

        for count in range(min_count, max_count + 1):
            yield from itertools.combinations_with_replacement(values, count)

    def n_combinations(self):
        """Get number of possible combinations with the current settings

        :return: number of possible combinations
        """

        count = 0

        # number of base resistors
        n_base_resistors = len(list(self._base_resistors()))

        # first count the base resistors
        if self.min_series < 2 or self.min_parallel < 2:
            count += n_base_resistors

        # add on series resistors
        if self.max_series >= 2:
            for i in range(self.min_series, self.max_series + 1):
                count += _n_comb_k(n_base_resistors, i, repetition=True)

        # add on parallel resistors
        if self.max_parallel >= 2:
            for i in range(self.min_parallel, self.max_parallel + 1):
                count += _n_comb_k(n_base_resistors, i, repetition=True)

        return count

    def closest(self, resistance, n_values=3, progress=True):
        """Returns closest resistors in set to target resistance

        :param resistance: target resistance
        :param n_values: number of resistors to match
        :param progress: show interactive progress bar
        :return: closest resistors
        """

        resistance = float(resistance)
        n_values = int(n_values)

        if resistance < 0:
            raise ValueError("Resistance must be > 0")

        # calculate number of results
        n_combinations = self.n_combinations()
        logging.getLogger("resistor").info("Calculating %i combinations",
                                            n_combinations)

        # generate combinations
        combinations = self.combinations()

        if progress:
            # add progress bar between voltage and heapq generators
            combinations = _print_progress(combinations, n_combinations)

        logging.getLogger("resistor").debug("Finding closest resistor matches")
        return heapq.nsmallest(n_values, combinations,
                               key=lambda i: abs(i.resistance - resistance))
