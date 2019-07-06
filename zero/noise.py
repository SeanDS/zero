"""Electronic noise sources"""

import abc
import numpy as np
from scipy.constants import Boltzmann

from .format import Quantity
from .components import BaseElement
from .config import ZeroConfig

CONF = ZeroConfig()


class NoiseNotFoundError(ValueError):
    def __init__(self, noise_description, *args, **kwargs):
        message = f"{noise_description} not found"
        super().__init__(message, *args, **kwargs)


class Noise(BaseElement, metaclass=abc.ABCMeta):
    """Noise source.

    Parameters
    ----------
    function : callable
        Callable that returns the noise associated with a specified frequency vector.
    component : :class:`Component`, optional
        Component associated with the noise. While optional, this must be set before the noise can
        be used in a calculation.
    """
    # Noise type, e.g. Johnson noise.
    NOISE_TYPE = None

    def __init__(self, function=None, component=None):
        super().__init__()
        self.function = function
        self.component = component

    def spectral_density(self, frequencies):
        return self.function(frequencies=frequencies)

    @property
    @abc.abstractmethod
    def label(self):
        return NotImplemented

    def _meta_data(self):
        """Meta data used to provide hash."""
        return tuple(self.label)

    @property
    def noise_type(self):
        return self.NOISE_TYPE

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self._meta_data())


class ComponentNoise(Noise, metaclass=abc.ABCMeta):
    """Component noise source."""
    ELEMENT_TYPE = "component"

    @property
    def component_type(self):
        return self.component.element_type


class NodeNoise(Noise, metaclass=abc.ABCMeta):
    """Node noise source.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the noise.
    """
    ELEMENT_TYPE = "node"

    def __init__(self, node=None, **kwargs):
        super().__init__(**kwargs)
        self.node = node


class VoltageNoise(ComponentNoise, metaclass=abc.ABCMeta):
    """Component voltage noise source."""
    NOISE_TYPE = "voltage"

    def __init__(self, **kwargs):
        super().__init__(function=self.noise_voltage, **kwargs)

    @abc.abstractmethod
    def noise_voltage(self, frequencies, **kwargs):
        raise NotImplementedError

    @property
    def label(self):
        return f"V({self.component.name})"


class OpAmpVoltageNoise(VoltageNoise):
    def noise_voltage(self, frequencies):
        return self.flat_noise * np.sqrt(1 + self.corner_frequency / frequencies)

    @property
    def flat_noise(self):
        return self.component.params["vnoise"]

    @property
    def corner_frequency(self):
        return self.component.params["vcorner"]


class ResistorJohnsonNoise(VoltageNoise):
    """Resistor Johnson-Nyquist noise source."""
    NOISE_TYPE = "johnson"

    def noise_voltage(self, frequencies):
        white_noise = np.sqrt(4 * Boltzmann * float(CONF["constants"]["T"]) * self.resistance)

        return np.ones_like(frequencies) * white_noise

    @property
    def resistance(self):
        return self.component.resistance

    @property
    def label(self):
        return f"R({self.component.name})"


class CurrentNoise(NodeNoise, metaclass=abc.ABCMeta):
    """Node current noise source."""
    NOISE_TYPE = "current"

    def __init__(self, **kwargs):
        super().__init__(function=self.noise_current, **kwargs)

    @abc.abstractmethod
    def noise_current(self, frequencies, **kwargs):
        raise NotImplementedError

    @property
    def label(self):
        return f"I({self.component.name}, {self.node.name})"


class OpAmpCurrentNoise(CurrentNoise):
    def noise_current(self, frequencies):
        # Ignore node; noise is same at both inputs.
        return self.flat_noise * np.sqrt(1 + self.corner_frequency / frequencies)

    @property
    def flat_noise(self):
        return self.component.params["inoise"]

    @property
    def corner_frequency(self):
        return self.component.params["icorner"]
