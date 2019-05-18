"""Data representation and manipulation"""

import abc
import logging
import numpy as np

from .config import ZeroConfig
from .misc import db

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()

def frequencies_match(vector_a, vector_b):
    return np.all(vector_a == vector_b)

def vectors_match(vector_a, vector_b):
    return np.allclose(vector_a, vector_b,
                       rtol=float(CONF["data"]["response_rel_tol"]),
                       atol=float(CONF["data"]["response_abs_tol"]))

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
            raise ValueError(f"cannot handle scale '{mag_scale}'")

        if phase_scale.lower() == "deg":
            phase = np.radians(phase)
        elif phase_scale.lower() == "rad":
            # don't need to scale
            pass
        else:
            raise ValueError(f"cannot handle scale '{phase_scale}'")

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

    def __mul__(self, other):
        if hasattr(other, "y"):
            # Extract data.
            other = other.y
        return Series(self.x, self.y * other)

    def __div__(self, other):
        if hasattr(other, "y"):
            # Extract data.
            other = other.y
        return Series(self.x, self.y / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __eq__(self, other):
        """Checks if the specified series is identical to this one, within tolerance"""
        return np.allclose(self.x, other.x) and np.allclose(self.y, other.y)


class Function(metaclass=abc.ABCMeta):
    """Data set"""
    def __init__(self, sources, sinks, series):
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series = series

    @property
    def frequencies(self):
        return self.series.x

    @abc.abstractmethod
    def draw(self, *axes):
        raise NotImplementedError

    @abc.abstractmethod
    def label(self, tex=False, suffix=None):
        raise NotImplementedError

    def __str__(self):
        return self.label()

    def meta_data(self):
        """Meta data used to provide a hash and check for meta equivalence."""
        return frozenset(self.sources), frozenset(self.sinks), self.label()

    def equivalent(self, other):
        """Checks if the specified function has equivalent sources, sinks, labels and data."""
        return self.meta_equivalent(other) and self.series_equivalent(other)

    def meta_equivalent(self, other):
        """Checks if the specified function has equivalent sources, sinks, and labels.

        This does not check for data equality.
        """
        return self.meta_data() == other.meta_data() \
               and frequencies_match(self.frequencies, other.frequencies)

    @abc.abstractmethod
    def series_equivalent(self, other):
        """Checks if the specified function has an equivalent series to this one."""
        raise NotImplementedError

    def __eq__(self, other):
        """Checks if the specified data set is identical to this one, within tolerance."""
        return self.equivalent(other)

    def __hash__(self):
        """Hash of the data set's meta data."""
        return hash(self.meta_data())


class SingleSourceFunction(Function, metaclass=abc.ABCMeta):
    """Data set containing data for a single source"""
    def __init__(self, source, **kwargs):
        # call parent constructor
        super().__init__(sources=[source], **kwargs)

    @property
    def source(self):
        return self.sources[0]

    @property
    def source_unit(self):
        return self.source.SOURCE_SINK_UNIT


class SingleSinkFunction(Function, metaclass=abc.ABCMeta):
    """Data set containing data for a single sink"""
    def __init__(self, sink, **kwargs):
        # call parent constructor
        super().__init__(sinks=[sink], **kwargs)

    @property
    def sink(self):
        return self.sinks[0]

    @property
    def sink_unit(self):
        return self.sink.SOURCE_SINK_UNIT


class Response(SingleSourceFunction, SingleSinkFunction, Function):
    """Response data series"""
    @property
    def magnitude(self):
        return db(np.abs(self.series.y))

    @property
    def phase(self):
        return np.angle(self.series.y) * 180 / np.pi

    def series_equivalent(self, other):
        """Checks if the specified function has an equivalent series to this one."""
        return vectors_match(self.magnitude, other.magnitude)

    def _draw_magnitude(self, axes, label_suffix=None):
        """Add magnitude plot to axes"""
        label = self.label(tex=True, suffix=label_suffix)
        axes.semilogx(self.frequencies, self.magnitude, label=label)

    def _draw_phase(self, axes):
        """Add phase plot to axes"""
        axes.semilogx(self.frequencies, self.phase)

    def draw(self, *axes, **kwargs):
        if len(axes) != 2:
            raise ValueError("two axes (magnitude and phase) must be provided")

        self._draw_magnitude(axes[0], **kwargs)
        self._draw_phase(axes[1])

    def label(self, tex=False, suffix=None):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$ (%s)%s"
        else:
            format_str = "%s to %s (%s)%s"

        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""

        return format_str % (self.source.label(), self.sink.label(), self.unit_str, suffix)

    @property
    def unit_str(self):
        if self.sink_unit is None and self.source_unit is None:
            return "dimensionless"
        if self.sink_unit is None:
            # Only source has unit.
            return f"1/{self.source_unit}"
        if self.source_unit is None:
            # Only sink has unit.
            return self.sink_unit

        # Both have units.
        return f"{self.sink_unit}/{self.source_unit}"


class NoiseDensityBase(SingleSinkFunction, metaclass=abc.ABCMeta):
    """Function with a single noise spectral density."""
    @property
    def spectral_density(self):
        return self.series.y

    def series_equivalent(self, other):
        """Checks if the specified function has an equivalent series to this one."""
        return spectra_match(self.spectral_density, other.spectral_density)

    def draw(self, *axes, label_suffix=None):
        if len(axes) != 1:
            raise ValueError("only one axis supported")

        axes = axes[0]

        label = self.label(tex=True, suffix=label_suffix)
        axes.loglog(self.frequencies, self.spectral_density, label=label)


class NoiseDensity(SingleSourceFunction, NoiseDensityBase):
    """Noise data series"""
    @property
    def noise_name(self):
        return str(self.source)

    @property
    def noise_type(self):
        return self.source.TYPE

    @property
    def noise_subtype(self):
        return self.source.SUBTYPE

    def label(self, tex=False, suffix=None):
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$%s"
        else:
            format_str = "%s to %s%s"

        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""

        return format_str % (self.noise_name, self.sink.label(), suffix)


class MultiNoiseDensity(NoiseDensityBase):
    """Set of noise data series from multiple sources to a single sink"""
    def __init__(self, sources=None, series=None, constituents=None, label="incoherent sum",
                 **kwargs):
        if series is None and constituents is None:
            raise ValueError("one of series or constituents must be specified")
        elif series is not None and constituents is not None:
            raise ValueError("only one of series and constituents can be specified")

        if series is not None:
            # only total noise is specified
            self.constituent_noise = None
        else:
            # derive series from constituents

            # sources are derived from constituents
            if sources is not None:
                raise ValueError("cannot specify constituents and sources together")

            # Use first spectral density to get sink and frequencies.
            frequencies = constituents[0].frequencies

            # check frequency axes are identical
            if not np.all([frequencies == spectral_density.series.x
                           for spectral_density in constituents]):
                raise ValueError("specified spectra do not share common x-axis")

            sources = [spectral_density.source for spectral_density in constituents]

            # store constituent series
            self.constituent_noise = [spectral_density.series for spectral_density in constituents]

            # create series
            noise_sum = np.sqrt(sum([data.y ** 2 for data in self.constituent_noise]))
            series = Series(frequencies, noise_sum)

        self._label = label

        # call parent constructor
        super().__init__(sources=sources, series=series, **kwargs)

        if constituents is not None:
            # check sink agrees with those set in constituents
            if not all([self.sink == spectral_density.sink for spectral_density in constituents]):
                raise Exception("cannot handle noise for functions with different sinks")

    @property
    def noise_names(self):
        if self.constituent_noise is None:
            return "unknown"

        return [spectral_density.noise_name for spectral_density in self.constituent_noise]

    def label(self, *args, suffix=None, **kwargs):
        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""
        return f"{self._label}{suffix}"
