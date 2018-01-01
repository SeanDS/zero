"""Data representation and manipulation"""

import abc
import numpy as np
import logging

from .config import ElectronicsConfig
from .misc import db

LOGGER = logging.getLogger("data")
CONF = ElectronicsConfig()

def frequencies_match(vector_a, vector_b):
    return np.all(vector_a == vector_b)

def magnitudes_match(vector_a, vector_b):
    return np.allclose(vector_a, vector_b,
                       rtol=float(CONF["data"]["magnitude_rel_tol"]),
                       atol=float(CONF["data"]["magnitude_abs_tol"]))

def phases_match(vector_a, vector_b):
    return np.allclose(vector_a, vector_b,
                       rtol=float(CONF["data"]["phase_rel_tol"]),
                       atol=float(CONF["data"]["phase_abs_tol"]))

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
        if magnitude_scale.lower() == "db":
            magnitude = 10 ** (magnitude / 20)
        else:
            raise Exception("cannot handle scale %s", magnitude_scale)

        if phase_scale.lower() == "degrees":
            phase = np.radians(phase)
        else:
            raise Exception("cannot handle scale %s", phase_scale)

        # convert magnitude and phase to complex
        complex_ = magnitude * (np.cos(phase) + np.sin(phase) * 1j)

        super(ComplexSeries, self).__init__(x=x, y=complex_)

class DataSet(object, metaclass=abc.ABCMeta):
    """Data set"""

    def __init__(self, sources, sinks, series_list):
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series_list = list(series_list)

    @abc.abstractmethod
    def draw(self, *axes):
        return NotImplemented

    @abc.abstractmethod
    def label(self, tex=False):
        return NotImplemented

    def __str__(self):
        return self.label()

class SingleDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing data from a single source to a single sink"""

    def __init__(self, source, sink, series):
        # call parent constructor
        super(SingleDataSet, self).__init__(sources=[source], sinks=[sink],
                                            series_list=[series])

    @property
    def source(self):
        return self.sources[0]

    @property
    def sink(self):
        return self.sinks[0]

    @property
    def series(self):
        return self.series_list[0]

    @series.setter
    def series(self, data):
        self.series_list = [data]

class TransferFunction(SingleDataSet, metaclass=abc.ABCMeta):
    """Transfer function data series"""

    @property
    def frequencies(self):
        return self.series.x

    @property
    def magnitude(self):
        return db(np.abs(self.series.y))

    @property
    def phase(self):
        return np.angle(self.series.y) * 180 / np.pi

    def _draw_magnitude(self, axes):
        """Add magnitude plot to axes"""

        axes.semilogx(self.frequencies, self.magnitude,
                      label=self.label(tex=True))

    def _draw_phase(self, axes):
        """Add phase plot to axes"""

        axes.semilogx(self.frequencies, self.phase)

    def draw(self, *axes):
        if len(axes) != 2:
            raise ValueError("two axes (magnitude and phase) must be provided")

        self._draw_magnitude(axes[0])
        self._draw_phase(axes[1])

    def label(self, tex=False):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$ %s"
        else:
            format_str = "%s to %s %s"

        return format_str % (self.source, self.sink,
                             self.unit_str)

    @property
    @abc.abstractmethod
    def unit_str(self):
        return NotImplemented

    def __eq__(self, other):
        if self.label() != other.label():
            return False
        elif not frequencies_match(self.frequencies, other.frequencies):
            LOGGER.error("%s frequencies don't match: %s != %s", self,
                         self.frequencies, other.frequencies)
            return False
        elif not magnitudes_match(self.magnitude, other.magnitude):
            LOGGER.error("%s magnitudes don't match: %s != %s", self,
                         self.magnitude, other.magnitude)
            return False
        elif not phases_match(self.phase, other.phase):
            LOGGER.error("%s phases don't match: %s != %s", self,
                         self.phase, other.phase)
            return False
        return True

class VoltageTransferFunction(TransferFunction):
    """Voltage transfer function data series"""

    @property
    def unit_str(self):
        return "(V/V)"

class CurrentTransferFunction(TransferFunction):
    """Current transfer function data series"""

    @property
    def unit_str(self):
        return "(A/V)"

class NoiseSpectrum(SingleDataSet):
    """Noise data series"""

    @property
    def frequencies(self):
        return self.series.x

    @property
    def spectrum(self):
        return self.series.y

    def draw(self, *axes):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]

        axes.loglog(self.frequencies, self.spectrum, label=self.label(tex=True))

    @property
    def noise_name(self):
        return str(self.source)

    def label(self, tex=False):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$"
        else:
            format_str = "%s to %s"

        return format_str % (self.noise_name, self.sink)

    def __eq__(self, other):
        if self.label() != other.label():
            return False
        elif not np.all(self.frequencies == other.frequencies):
            return False
        elif not np.allclose(self.spectrum, other.spectrum):
            return False
        return True

class MultiNoiseSpectrum(DataSet):
    """Noise data series from multiple sources to a single sink"""

    def __init__(self, spectra, *args, **kwargs):
        # use first spectrum to get sink and frequencies
        sink = spectra[0].sink
        frequencies = spectra[0].series.x

        # check frequency axes are identical
        if not np.all([frequencies == spectrum.series.x for spectrum in spectra]):
            raise ValueError("specified spectra do not share same x-axis")

        # check sinks are identical
        if not all([sink == spectrum.sink for spectrum in spectra]):
            raise Exception("cannot plot total noise for functions with "
                            "different sinks")

        sources = [spectrum.source for spectrum in spectra]
        series = [spectrum.series for spectrum in spectra]

        # call parent constructor
        super(MultiNoiseSpectrum, self).__init__(sources=sources, sinks=[sink],
                                                 series_list=series, *args,
                                                 **kwargs)

        self.noise_names = [spectrum.noise_name for spectrum in spectra]

    def draw(self, *axes):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]

        axes.loglog(self.series.x, self.series.y, label=self.label(tex=True))

    @property
    def series(self):
        x = self.frequencies
        y = np.sqrt(sum([data.y ** 2 for data in self.series_list]))

        return Series(x, y)

    @property
    def frequencies(self):
        # use first series
        return self.series_list[0].x

    @property
    def sink(self):
        return self.sinks[0]

    def label(self, *args):
        return "incoherent sum"
