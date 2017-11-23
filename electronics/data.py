"""Data representation and manipulation"""

import abc
import numpy as np

from .misc import db

class Series(object):
    """Data series"""

    SCALE_DB = 1
    SCALE_DEG = 2

    def __init__(self, x, y):
        self.x = x
        self.y = y

class ComplexSeries(Series):
    """Complex data series"""

    def __init__(self, x, magnitude, phase, magnitude_scale, phase_scale):
        if magnitude_scale == "db":
            magnitude = 10 ** (magnitude / 20)
        else:
            raise Exception("cannot handle scale %s", magnitude_scale)

        # convert magnitude and phase to complex
        complex = magnitude * (np.cos(phase) + np.sin(phase) * 1j)

        super(ComplexSeries, self).__init__(x=x, y=complex)

class DataSet(object, metaclass=abc.ABCMeta):
    """Data set"""

    def __init__(self, source, sink, series):
        self.source = source
        self.sink = sink
        self.series = series

    @abc.abstractmethod
    def draw(self, *axes):
        return NotImplemented

    @property
    @abc.abstractmethod
    def label(self):
        return NotImplemented

class TransferFunction(DataSet):
    """Transfer function data series"""

    def __init__(self, *args, **kwargs):
        # call parent constructor
        super(TransferFunction, self).__init__(*args, **kwargs)

    def _draw_magnitude(self, axes):
        """Add magnitude plot to axes"""

        axes.semilogx(self.series.x, db(np.abs(self.series.y)),
                      label=self.label)

    def _draw_phase(self, axes):
        """Add phase plot to axes"""

        axes.semilogx(self.series.x, np.angle(self.series.y) * 180 / np.pi)

    def draw(self, *axes):
        if len(axes) != 2:
            raise ValueError("two axes (magnitude and phase) must be provided")

        self._draw_magnitude(axes[0])
        self._draw_phase(axes[1])

    @property
    def label(self):
        return "%s to %s" % (self.source, self.sink)

class NoiseSpectrum(DataSet):
    """Noise data series"""

    NOISE_JOHNSON = 1
    NOISE_OPAMP_VOLTAGE = 2
    NOISE_OPAMP_CURRENT = 3

    NAMES = {NOISE_JOHNSON: "Johnson",
             NOISE_OPAMP_VOLTAGE: "Op-amp voltage",
             NOISE_OPAMP_CURRENT: "Op-amp current"}

    def __init__(self, noise_type, *args, **kwargs):
        # call parent constructor
        super(NoiseSpectrum, self).__init__(*args, **kwargs)

        self.noise_type = noise_type

    def draw(self, *axes):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]

        axes.loglog(self.series.x, self.series.y, label=self.label)

    @property
    def noise_name(self):
        return self.NAMES[self.noise_type]

    @property
    def label(self):
        return "%s noise %s to %s" % (self.noise_name, self.source, self.sink)
