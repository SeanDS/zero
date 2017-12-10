"""Electronic components"""

import abc
import numpy as np

from .format import SIFormatter
from .config import ElectronicsConfig, OpAmpLibrary

CONF = ElectronicsConfig()
LIBRARY = OpAmpLibrary()

class BaseComponent(object, metaclass=abc.ABCMeta):
    """Class representing a component"""

    @abc.abstractmethod
    def noise_voltage(self, frequency):
        """Noise voltage in V/sqrt(Hz)

        :param frequency: frequency in Hz
        :type frequency: float or Numpy scalar
        :return: noise voltage
        :rtype: float or Numpy scalar
        """

        return NotImplemented

    @property
    @abc.abstractmethod
    def label(self):
        return NotImplemented

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.label

class PassiveComponent(BaseComponent, metaclass=abc.ABCMeta):
    """Represents a passive component with a defined value and impedance"""

    UNIT = "?"

    def __init__(self, value, tolerance=None, *args, **kwargs):
        super(PassiveComponent, self).__init__(*args, **kwargs)

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

    @abc.abstractmethod
    def impedance(self, frequency):
        return NotImplemented

    @property
    def label(self):
        """Label containing name and formatted value

        :return: label
        :rtype: str
        """

        value = SIFormatter.format(self.value, self.UNIT)

        if self.tolerance:
            # append tolerance to value
            value += " ± {}%".format(SIFormatter.format(self.tolerance))

        return value

    def __float__(self):
        """Component float equivalent is their value"""
        return self.value

class Resistor(PassiveComponent):
    """Represents a resistor or set of series or parallel resistors"""

    UNIT = "Ω"

    # tolerances, in percent (+/-)
    TOL_NONE = None
    TOL_GREY = 0.05
    TOL_VIOLET = 0.1
    TOL_BLUE = 0.25
    TOL_GREEN = 0.5
    TOL_BROWN = 1.0
    TOL_RED = 2.0
    TOL_GOLD = 5.0
    TOL_SILVER = 10.0

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

    def noise_voltage(self, *args):
        # no noise voltage in perfect capacitor
        return 0

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

    def noise_voltage(self, *args):
        # no noise voltage in perfect inductor
        return 0

class OpAmp(BaseComponent):
    """Represents an (almost) ideal op-amp"""

    def __init__(self, model=None, *args, **kwargs):
        # call parent constructor
        super(OpAmp, self).__init__(*args, **kwargs)

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

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        model = str(model).upper()

        if not LIBRARY.has_data(model):
            raise ValueError("unrecognised op-amp type: %s" % model)

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

    def gain(self, frequency):
        return (self._a0 / (1 + self._a0 * 1j * frequency / self._gbw)
                * np.exp(-1j * 2 * np.pi * self._delay * frequency)
                * np.prod(1 + 1j * frequency / self._zeros)
                / np.prod(1 + 1j * frequency / self._poles))

    def inverse_gain(self, *args, **kwargs):
        return 1 / self.gain(*args, **kwargs)

    def noise_voltage(self, frequency):
        return self._vn * np.sqrt(1 + self._vc / frequency)

    @property
    def label(self):
        return self.model
