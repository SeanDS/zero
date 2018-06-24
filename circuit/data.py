"""Data representation and manipulation"""

import abc
import numpy as np
import logging

from .config import CircuitConfig
from .misc import db

LOGGER = logging.getLogger("data")
CONF = CircuitConfig()

def frequencies_match(vector_a, vector_b):
    return np.all(vector_a == vector_b)

def tfs_match(vector_a, vector_b):
    return np.allclose(vector_a, vector_b,
                       rtol=float(CONF["data"]["tf_rel_tol"]),
                       atol=float(CONF["data"]["tf_abs_tol"]))

def spectra_match(vector_a, vector_b):
    return np.allclose(vector_a, vector_b,
                       rtol=float(CONF["data"]["noise_rel_tol"]),
                       atol=float(CONF["data"]["noise_abs_tol"]))

def argmax_difference(vector_a, vector_b):
    """Finds the maximum relative difference in percent between `vector_a` and `vector_b`
    
    Returns index of maximum relative difference as well as its value.
    """

    # relative difference in percent between `vector_a` and `vector_b`
    difference = 100 * np.abs(vector_a - vector_b) / np.abs(vector_a)

    # index of maximum difference
    i = np.argmax(difference)

    return i, difference[i]

class Series(object):
    """Data series"""

    SCALE_DB = 1
    SCALE_DEG = 2

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, factor):
        return Series(self.x, self.y * factor)

class ComplexSeries(Series):
    """Complex data series"""
    def __init__(self, x, magnitude, phase, magnitude_scale, phase_scale):
        if magnitude_scale.lower() == "db":
            magnitude = 10 ** (magnitude / 20)
        elif magnitude_scale.lower() == "abs":
            # don't need to scale
            pass
        else:
            raise Exception("cannot handle scale %s" % magnitude_scale)

        if phase_scale.lower() == "degrees":
            phase = np.radians(phase)
        else:
            raise Exception("cannot handle scale %s" % phase_scale)

        # convert magnitude and phase to complex
        complex_ = magnitude * (np.cos(phase) + np.sin(phase) * 1j)

        super().__init__(x=x, y=complex_)

class DataSet(object, metaclass=abc.ABCMeta):
    """Data set"""
    def __init__(self, sources, sinks, series_list):
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series_list = list(series_list)

    @abc.abstractmethod
    def draw(self, *axes):
        raise NotImplementedError

    @abc.abstractmethod
    def label(self, tex=False):
        return NotImplemented

    def __str__(self):
        return self.label()

class SingleSeriesDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing a single data series"""
    def __init__(self, series, *args, **kwargs):
        # call parent constructor
        super().__init__(series_list=[series], *args, **kwargs)

    @property
    def series(self):
        return self.series_list[0]

    @series.setter
    def series(self, data):
        self.series_list = [data]

class SingleSourceDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing data for a single source"""
    def __init__(self, source, *args, **kwargs):
        # call parent constructor
        super().__init__(sources=[source], *args, **kwargs)

    @property
    def source(self):
        return self.sources[0]

class SingleSinkDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing data for a single sink"""
    def __init__(self, sink, *args, **kwargs):
        # call parent constructor
        super().__init__(sinks=[sink], *args, **kwargs)

    @property
    def sink(self):
        return self.sinks[0]

class TransferFunction(SingleSeriesDataSet, SingleSourceDataSet, SingleSinkDataSet, metaclass=abc.ABCMeta):
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

        return format_str % (self.source.label(), self.sink.label(), self.unit_str)

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
        elif not tfs_match(self.magnitude, other.magnitude):
            # calculate worst relative difference between tfs
            worst_i, worst_diff = argmax_difference(self.magnitude, other.magnitude)

            LOGGER.error("%s tf magnitudes don't match (worst difference %f%% at %d (%f, %f))",
                         self, worst_diff, self.frequencies[worst_i], self.magnitude[worst_i],
                         other.magnitude[worst_i])
            return False
        return True

class VoltageVoltageTF(TransferFunction):
    """Voltage to voltage transfer function data series"""

    @property
    def unit_str(self):
        return "(V/V)"

class VoltageCurrentTF(TransferFunction):
    """Voltage to current transfer function data series"""

    @property
    def unit_str(self):
        return "(A/V)"

class CurrentCurrentTF(TransferFunction):
    """Current to current transfer function data series"""

    @property
    def unit_str(self):
        return "(A/A)"

class CurrentVoltageTF(TransferFunction):
    """Current to voltage transfer function data series"""

    @property
    def unit_str(self):
        return "(V/A)"

class NoiseSpectrum(SingleSeriesDataSet, SingleSourceDataSet, SingleSinkDataSet):
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
        elif not frequencies_match(self.frequencies, other.frequencies):
            return False
        elif not spectra_match(self.spectrum, other.spectrum):
            return False
        return True

class MultiNoiseSpectrum(SingleSinkDataSet):
    """Set of noise data series from multiple sources to a single sink"""
    def __init__(self, spectra, label="incoherent sum", *args, **kwargs):
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
        super().__init__(sources=sources, sink=sink, series_list=series, *args, **kwargs)

        self.noise_names = [spectrum.noise_name for spectrum in spectra]
        self._label = label

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

    def label(self, *args, **kwargs):
        return self._label

class SumNoiseSpectrum(SingleSeriesDataSet, SingleSinkDataSet):
    """Single sum noise data series from multiple sources to a single sink"""
    def draw(self, *axes):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]
        axes.loglog(self.series.x, self.series.y, label=self.label(tex=True))

    @property
    def frequencies(self):
        # use first series
        return self.series_list[0].x

    def label(self, *args, **kwargs):
        return "sum"
