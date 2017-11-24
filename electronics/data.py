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

    def __init__(self, sources, sinks, series_list):
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series_list = list(series_list)

    @abc.abstractmethod
    def draw(self, *axes):
        return NotImplemented

    @property
    @abc.abstractmethod
    def label(self):
        return NotImplemented

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

class TransferFunction(SingleDataSet):
    """Transfer function data series"""

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

class NoiseSpectrum(SingleDataSet):
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

    def draw(self, *axes):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]

        axes.loglog(self.series.x, self.series.y, label=self.label)

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

    @property
    def label(self):
        sources = ", ".join(self.sources)
        return "total noise %s to %s" % (sources, self.sink)
