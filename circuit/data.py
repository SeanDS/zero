"""Data representation and manipulation"""

import abc
import logging
import numpy as np

from .config import CircuitConfig
from .misc import db

LOGGER = logging.getLogger(__name__)
CONF = CircuitConfig()

def frequencies_match(vector_a, vector_b):
    return np.all(vector_a == vector_b)

def vectors_match(vector_a, vector_b):
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


class Series:
    """Data series"""
    def __init__(self, x, y):
        if x.shape != y.shape:
            raise ValueError("specified x and y vectors do not have the same shape")

        self.x = x
        self.y = y

    @classmethod
    def from_mag_phase(cls, x, magnitude, phase=None, mag_scale=None, phase_scale=None):
        """Create :class:`Series` from magnitude and phase data.

        Parameters
        ----------
        x : :class:`np.array`
            The x vector.
        magnitude : :class:`np.array`
            The magnitude.
        phase : :class:`np.array`, optional
            The phase. If `None`, the magnitude is assumed to have zero phase.
        mag_scale : :class:`str`, optional
            The magnitude scale. Defaults to absolute.
        phase_scale : :class:`str`, optional
            The phase scale. Defaults to degrees.

        Returns
        -------
        :class:`Series`
            The series containing the data.

        Raises
        ------
        :class:`ValueError`
            If the specified magnitude or phase scale is unrecognised.
        """
        if phase is None:
            # set phase to zero
            phase = np.zeros_like(magnitude)
        if mag_scale is None:
            mag_scale = "abs"
        if phase_scale is None:
            phase_scale = "deg"

        if mag_scale.lower() == "db":
            magnitude = 10 ** (magnitude / 20)
        elif mag_scale.lower() == "abs":
            # don't need to scale
            pass
        else:
            raise ValueError("cannot handle scale %s" % mag_scale)

        if phase_scale.lower() == "deg":
            phase = np.radians(phase)
        elif phase_scale.lower() == "rad":
            # don't need to scale
            pass
        else:
            raise ValueError("cannot handle scale %s" % phase_scale)

        # convert magnitude and phase to complex
        complex_ = magnitude * (np.cos(phase) + 1j * np.sin(phase))

        return cls(x=x, y=complex_)

    @classmethod
    def from_re_im(cls, x, re, im):
        """Create :class:`Series` from real and imaginary parts.

        Parameters
        ----------
        x : :class:`np.array`
            The x vector.
        magnitude : :class:`np.array`
            The magnitude.
        phase : :class:`np.array`
            The phase.
        magnitude_scale : :class:`str`, optional
            The magnitude scale. Defaults to absolute.
        phase_scale : :class:`str`, optional
            The phase scale. Defaults to radians.

        Returns
        -------
        :class:`Series`
            The series containing the data.

        Raises
        ------
        :class:`ValueError`
            If either the real or imaginary part is complex.
        """
        if np.any(np.iscomplex(re)) or np.any(np.iscomplex(im)):
            raise ValueError("specified real and imaginary parts must not be complex")

        # combine into complex
        complex_ = re + 1j * im

        return cls(x=x, y=complex_)

    def __mul__(self, factor):
        return Series(self.x, self.y * factor)


class DataSet(metaclass=abc.ABCMeta):
    """Data set"""
    def __init__(self, sources, sinks, series_list):
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series_list = list(series_list)

        # text to add in brackets after label
        self.label_suffix = ""

    @abc.abstractmethod
    def draw(self, *axes):
        raise NotImplementedError

    def label(self, tex=False):
        label_str = self._label_base(tex)

        if self.label_suffix:
            label_str += " (%s)" % self.label_suffix

        return label_str

    @abc.abstractmethod
    def _label_base(self, tex):
        """Data set label, without suffix."""
        raise NotImplementedError

    def matches(self, other):
        if self.label() != other.label():
            return False

        return True

    def __str__(self):
        return self.label()

    def __eq__(self, other):
        """Equality operator.

        Note: DataSet objects are considered equal if they have equal sources, sinks, lists of
        series, and labels.
        """
        if self.sources != other.sources:
            return False
        if self.sinks != other.sinks:
            return False
        if self.series_list != other.series_list:
            return False
        if self.label() != other.label():
            return False

        return True

    def __hash__(self):
        return hash((tuple(self.sources), tuple(self.sinks), tuple(self.series_list), self.label()))


class SingleSeriesDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing a single data series"""
    def __init__(self, series, **kwargs):
        # call parent constructor
        super().__init__(series_list=[series], **kwargs)

    @property
    def series(self):
        return self.series_list[0]

    @series.setter
    def series(self, data):
        self.series_list = [data]

    @property
    def frequencies(self):
        return self.series.x

    def matches(self, other):
        if not super().matches(other):
            return False

        if not frequencies_match(self.frequencies, other.frequencies):
            return False

        return True


class SingleSourceDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing data for a single source"""
    def __init__(self, source, source_unit=None, **kwargs):
        # call parent constructor
        super().__init__(sources=[source], **kwargs)

        self.source_unit = source_unit

    @property
    def source(self):
        return self.sources[0]


class SingleSinkDataSet(DataSet, metaclass=abc.ABCMeta):
    """Data set containing data for a single sink"""
    def __init__(self, sink, sink_unit=None, **kwargs):
        # call parent constructor
        super().__init__(sinks=[sink], **kwargs)

        self.sink_unit = sink_unit

    @property
    def sink(self):
        return self.sinks[0]


class TransferFunction(SingleSeriesDataSet, SingleSourceDataSet, SingleSinkDataSet):
    """Transfer function data series"""
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

    def _label_base(self, tex=False):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$ (%s)"
        else:
            format_str = "%s to %s (%s)"

        return format_str % (self.source.label(), self.sink.label(), self.unit_str)

    @property
    def unit_str(self):
        if self.sink_unit is None and self.source_unit is None:
            return "dimensionless"
        elif self.sink_unit is None:
            # only source has unit
            return "1/%s" % self.source_unit
        elif self.source_unit is None:
            # only sink has unit
            return self.sink_unit

        # both have units
        return "%s/%s" % (self.sink_unit, self.source_unit)

    def matches(self, other):
        if not super().matches(other):
            return False

        if not vectors_match(self.magnitude, other.magnitude):
            # calculate worst relative difference between tfs
            worst_i, worst_diff = argmax_difference(self.magnitude, other.magnitude)

            LOGGER.error("%s tf magnitudes don't match (worst difference %f%% at %d (%f, %f))",
                         self, worst_diff, self.frequencies[worst_i], self.magnitude[worst_i],
                         other.magnitude[worst_i])
            return False

        return True


class NoiseSpectrum(SingleSeriesDataSet, SingleSourceDataSet, SingleSinkDataSet):
    """Noise data series"""
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

    @property
    def noise_type(self):
        return self.source.TYPE

    @property
    def noise_subtype(self):
        return self.source.SUBTYPE

    def _label_base(self, tex=False):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$"
        else:
            format_str = "%s to %s"

        return format_str % (self.noise_name, self.sink)

    def matches(self, other):
        if not super().matches(other):
            return False

        if not spectra_match(self.spectrum, other.spectrum):
            return False

        return True


class MultiNoiseSpectrum(SingleSinkDataSet):
    """Set of noise data series from multiple sources to a single sink"""
    def __init__(self, spectra, label="incoherent sum", **kwargs):
        spectra = list(spectra)

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
        super().__init__(sources=sources, sink=sink, series_list=series, **kwargs)

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

    @property
    def spectrum(self):
        return self.series.y

    def _label_base(self, *args, **kwargs):
        return self._label

    def matches(self, other):
        if not super().matches(other):
            return False

        if not spectra_match(self.spectrum, other.spectrum):
            return False

        return True


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
        return self.series.x

    @property
    def spectrum(self):
        return self.series.y

    def _label_base(self, *args, **kwargs):
        return "incoherent sum"

    def matches(self, other):
        if not super().matches(other):
            return False

        if not spectra_match(self.spectrum, other.spectrum):
            return False

        return True
