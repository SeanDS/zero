"""Electronic noise sources"""

import abc
import numpy as np

from .components import BaseElement
from .config import ZeroConfig

CONF = ZeroConfig()


class Noise(BaseElement, metaclass=abc.ABCMeta):
    """Noise source.

    Parameters
    ----------
    function : callable
        Callable that returns the noise associated with a specified frequency vector.
    """
    # Noise type, e.g. Johnson noise.
    NOISE_TYPE = None

    def __init__(self, function=None):
        super().__init__()
        self.function = function

    @abc.abstractmethod
    def spectral_density(self, frequencies):
        return NotImplemented

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
    """Component noise source.

    Parameters
    ----------
    component : :class:`Component`
        Component associated with the noise.
    """
    ELEMENT_TYPE = "component"

    def __init__(self, component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.component = component

    def spectral_density(self, frequencies):
        return self.function(component=self.component, frequencies=frequencies)

    def _meta_data(self):
        """Meta data used to provide hash."""
        return super()._meta_data(), self.component

    @property
    def component_type(self):
        return self.component.element_type


class NodeNoise(Noise, metaclass=abc.ABCMeta):
    """Node noise source.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the noise.
    component : :class:`Component`
        Component associated with the noise.
    """
    ELEMENT_TYPE = "node"

    def __init__(self, node, component, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = node
        self.component = component

    def spectral_density(self, *args, **kwargs):
        return self.function(node=self.node, *args, **kwargs)

    def _meta_data(self):
        """Meta data used to provide hash."""
        return super()._meta_data(), self.node, self.component


class VoltageNoise(ComponentNoise):
    """Component voltage noise source."""
    NOISE_TYPE = "voltage"

    @property
    def label(self):
        return f"V({self.component.name})"


class JohnsonNoise(VoltageNoise):
    """Resistor Johnson-Nyquist noise source."""
    NOISE_TYPE = "johnson"

    def __init__(self, *args, **kwargs):
        super().__init__(function=self.noise_voltage, *args, **kwargs)

    def noise_voltage(self, frequencies, *args, **kwargs):
        white_noise = np.sqrt(4 * float(CONF["constants"]["kB"])
                              * float(CONF["constants"]["T"])
                              * self.resistance)

        return np.ones_like(frequencies) * white_noise

    @property
    def resistance(self):
        return self.component.resistance

    @property
    def label(self):
        return f"R({self.component.name})"

    def _meta_data(self):
        """Meta data used to provide hash."""
        return super()._meta_data(), self.resistance


class CurrentNoise(NodeNoise):
    """Node current noise source."""
    NOISE_TYPE = "current"

    @property
    def label(self):
        return f"I({self.component.name}, {self.node.name})"


class NoiseNotFoundError(ValueError):
    def __init__(self, noise_description, *args, **kwargs):
        message = f"{noise_description} not found"
        super().__init__(message, *args, **kwargs)
