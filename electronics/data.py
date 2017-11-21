"""Data representation and manipulation"""

import abc
import numpy as np

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
        # TODO: handle scales
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
    def description(self):
        return NotImplemented

class TransferFunction(DataSet):
    """Transfer function data series"""

    def __init__(self, *args, **kwargs):
        # call parent constructor
        super(TransferFunction, self).__init__(*args, **kwargs)

    def description(self):
        return "transfer function from %s to %s" % (self.source, self.sink)

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

    @property
    def noise_name(self):
        return self.NAMES[self.noise_type]

    def description(self):
        return "%s noise at %s from %s" % (self.noise_name, self.sink,
                                           self.source)
