"""Data representation and manipulation"""

import abc
import logging
from numbers import Number
import numpy as np

from .config import ZeroConfig
from .misc import mag_to_db, db_to_mag

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
            The magnitude. This magnitude's scaling is determined by `mag_scale`.
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
            magnitude = db_to_mag(magnitude)
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
            The magnitude. This magnitude's scaling is determined by `mag_scale`.
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

    def __add__(self, other):
        other = getattr(other, "y", other)
        return self._new_series(self.y + other)

    def __sub__(self, other):
        return self + -other

    def __neg__(self):
        return self._new_series(-self.y)

    def __mul__(self, other):
        other = getattr(other, "y", other)
        return self._new_series(self.y * other)

    def __rmul__(self, other):
        # Series multiplication is commutative.
        return self * other

    def __truediv__(self, other):
        if hasattr(other, "y"):
            return self * other.inverse()
        return self._new_series(self.y * 1 / other)

    def __rtruediv__(self, other):
        return other * self.inverse()

    def __pow__(self, other):
        other = getattr(other, "y", other)
        return self._new_series(self.y ** other)

    def inverse(self):
        return self._new_series(np.reciprocal(self.y))

    def _new_series(self, new_y):
        return self.__class__(self.x, new_y)

    def __eq__(self, other):
        """Checks if the specified series is identical to this one, within tolerance"""
        return np.allclose(self.x, other.x) and np.allclose(self.y, other.y)


class BaseFunction(metaclass=abc.ABCMeta):
    """Base function container.

    A function represents data between one or many sources and sinks. These can be any type
    descending from :class:`.BaseElement`, though concrete subclasses may implement additional type
    constraints.

    Functions are designed to allow mathematical operations, such as multiplication by scalars or
    other functions. Concrete subclasses may implement additional constraints on allowed operations.

    Parameters
    ----------
    sources, sinks : list of :class:`.BaseElement`, optional
        The function's sources and sinks. Defaults to empty lists.
    series : :class:`.Series`, optional
        The function's data.
    plot_options : :class:`dict`, optional
        Plot options, passed to :meth:`.matplotlib.pyplot.plot`.
    """
    def __init__(self, sources=None, sinks=None, series=None, plot_options=None):
        self._label = None
        if sources is None:
            sources = []
        if sinks is None:
            sinks = []
        if plot_options is None:
            plot_options = {}
        self.sources = list(sources)
        self.sinks = list(sinks)
        self.series = series
        self.plot_options = dict(plot_options)

        self._set_fallback_plot_options()

    def _set_fallback_plot_options(self):
        """Set plot options in cases where user-specified options are not provided."""
        if "alpha" not in self.plot_options:
            self._set_plot_option("alpha", CONF["plot"]["alpha"])
        if "dash_capstyle" not in self.plot_options:
            self._set_plot_option("dash_capstyle", CONF["plot"]["dash_capstyle"])
        if "linestyle" not in self.plot_options and "ls" not in self.plot_options:
            self._set_plot_option("linestyle", CONF["plot"]["linestyle"])
        if "linewidth" not in self.plot_options and "lw" not in self.plot_options:
            self._set_plot_option("linewidth", CONF["plot"]["linewidth"])
        if "zorder" not in self.plot_options:
            self._set_plot_option("zorder", CONF["plot"]["zorder"])

    def _set_plot_option(self, key, value):
        if value is None:
            # Ignore empty values, to allow the user to ignore options in the configuration.
            return
        self.plot_options[key] = value

    @property
    def frequencies(self):
        return self.series.x

    @abc.abstractmethod
    def draw(self, *axes):
        raise NotImplementedError

    @property
    def label(self):
        return self._format_label()

    @label.setter
    def label(self, label):
        self._label = label

    @abc.abstractmethod
    def _format_label(self, tex=False, suffix=None, ignore_user_label=False):
        raise NotImplementedError

    def __str__(self):
        return self.label

    def meta_data(self):
        """Meta data used to provide a hash and check for meta equivalence."""
        return frozenset(self.sources), frozenset(self.sinks), self.label

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


class SingleSourceFunction(BaseFunction, metaclass=abc.ABCMeta):
    """Data set containing data for a single source."""
    def __init__(self, source=None, **kwargs):
        if source is not None:
            sources = [source]
        else:
            sources = []
        super().__init__(sources=sources, **kwargs)

    @property
    def source(self):
        return self.sources[0]

    @source.setter
    def source(self, source):
        self.sources[0] = source

    @property
    def source_unit(self):
        return self.source.element_unit


class SingleSinkFunction(BaseFunction, metaclass=abc.ABCMeta):
    """Data set containing data for a single sink."""
    def __init__(self, sink=None, **kwargs):
        if sink is not None:
            sinks = [sink]
        else:
            sinks = []
        super().__init__(sinks=sinks, **kwargs)

    @property
    def sink(self):
        return self.sinks[0]

    @sink.setter
    def sink(self, sink):
        self.sinks[0] = sink

    @property
    def sink_unit(self):
        return self.sink.element_unit


class Response(SingleSourceFunction, SingleSinkFunction):
    """Data set representing a response at a sink from a source."""
    @property
    def complex_magnitude(self):
        return self.series.y

    @property
    def magnitude(self):
        """Absolute magnitude."""
        return np.abs(self.complex_magnitude)

    @property
    def db_magnitude(self):
        r"""Magnitude scaled in units of decibel.

        The response is power scaled such that the response is :math:`20 \log_{10} \left| x \right|`
        where :math:`x` is the complex response provided by :attr:`.complex_magnitude`.
        """
        return mag_to_db(self.magnitude)

    @property
    def phase(self):
        """Phase in degrees."""
        return np.angle(self.complex_magnitude) * 180 / np.pi

    def series_equivalent(self, other):
        """Checks if the specified function has an equivalent series to this one."""
        return vectors_match(self.magnitude, other.magnitude)

    def _draw_magnitude(self, axes, label_suffix=None, scale_db=True):
        """Add magnitude plot to axes"""
        label = self._format_label(tex=True, suffix=label_suffix)
        if scale_db:
            # Decibel y-axis scaling.
            axes.semilogx(self.frequencies, self.db_magnitude, label=label, **self.plot_options)
        else:
            # Linear y-axis scaling.
            axes.loglog(self.frequencies, self.magnitude, label=label, **self.plot_options)

    def _draw_phase(self, axes):
        """Add phase plot to axes"""
        axes.semilogx(self.frequencies, self.phase, **self.plot_options)

    def draw(self, *axes, **kwargs):
        if len(axes) != 2:
            raise ValueError("two axes (magnitude and phase) must be provided")

        self._draw_magnitude(axes[0], **kwargs)
        self._draw_phase(axes[1])

    def _format_label(self, tex=False, suffix=None, ignore_user_label=False):
        if not ignore_user_label and self._label is not None:
            return self._label
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$ (%s)%s"
        else:
            format_str = "%s to %s (%s)%s"

        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""

        return format_str % (self.source.label, self.sink.label, self.unit_str, suffix)

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

    def __mul__(self, other):
        if isinstance(other, Response):
            other_sink = other.sink
            other_value = other.series
            if self.sink_unit != other.source_unit:
                raise ValueError(f"Cannot multiply this response by {other}: the sink unit of this "
                                 f"response, {self.sink_unit}, is incompatible with the source "
                                 f"unit of {other}, {other.source_unit}.")
        elif isinstance(other, Number):
            other_sink = self.sink
            other_value = other
        else:
            # E.g. NoiseDensityBase.
            raise TypeError(f"Cannot multiply {self.__class__.__name__} by "
                            f"{other.__class__.__name__}.")
        return self._new_response(other_sink, self.series * other_value)

    def __rmul__(self, other):
        # Note: "other" should never be another Response or NoiseDensityBase, since these functions
        # implement the corresponding __mul__ methods for Response.
        if not isinstance(other, Number):
            raise TypeError(f"Cannot multiply {other.__class__.__name__} by "
                            f"{self.__class__.__name__}.")
        # Response-scalar multiplication is commutative.
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f"Cannot divide {self.__class__.__name__} by "
                            f"{other.__class__.__name__}. To invert, call inverse().")
        return self._new_response(self.sink, self.series * 1 / other)

    def __rtruediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f"Cannot divide {other.__class__.__name__} by "
                            f"{self.__class__.__name__}. To invert, call inverse().")
        return self._new_response(self.sink, other / self.series)

    def inverse(self):
        """Inverse response."""
        return self.__class__(source=self.sink, sink=self.source, series=self.series.inverse())

    def _new_response(self, new_sink, new_series):
        """Create a copy of this response with the specified sink and series."""
        new_response = self.__class__(source=self.source, sink=new_sink, series=new_series)
        new_response.label = self._label
        return new_response


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

        label = self._format_label(tex=True, suffix=label_suffix)
        axes.loglog(self.frequencies, self.spectral_density, label=label, **self.plot_options)

    def __mul__(self, other):
        if isinstance(other, Response):
            other_sink = other.sink
            other_value = other.magnitude
            if self.sink_unit != other.source_unit:
                raise ValueError(f"Cannot multiply this noise by {other}: the sink unit of this "
                                 f"noise, {self.sink_unit}, is incompatible with the source unit "
                                 f"of {other}, {other.source_unit}.")
        elif isinstance(other, Number):
            other_sink = self.sink
            other_value = other
        else:
            # E.g. NoiseDensityBase.
            raise TypeError(f"Cannot multiply {self.__class__.__name__} by "
                            f"{other.__class__.__name__}.")
        return self._new_noise_density(other_sink, self.series * other_value)

    def __rmul__(self, other):
        # Note: "other" should never be another Response or NoiseDensityBase, since these functions
        # implement the corresponding __mul__ methods for NoiseDensityBase.
        if not isinstance(other, Number):
            raise TypeError(f"Cannot divide {other.__class__.__name__} by "
                            f"{self.__class__.__name__}.")
        # Noise-scalar multiplication is commutative.
        return self * other

    @abc.abstractmethod
    def _new_noise_density(self, new_sink, new_series):
        raise NotImplementedError


class NoiseDensity(SingleSourceFunction, NoiseDensityBase):
    """Noise data series"""
    @property
    def noise_name(self):
        return str(self.source)

    @property
    def element_type(self):
        return self.source.element_type

    @property
    def noise_type(self):
        return self.source.noise_type

    def _format_label(self, tex=False, suffix=None, ignore_user_label=False):
        if not ignore_user_label and self._label is not None:
            return self._label
        if tex:
            format_str = r"$\bf{%s}$ to $\bf{%s}$%s"
        else:
            format_str = "%s to %s%s"

        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""

        return format_str % (self.noise_name, self.sink.label, suffix)

    def _new_noise_density(self, new_sink, new_series):
        """Create a copy of this noise density with the specified sink and series."""
        new_noise = self.__class__(source=self.source, sink=new_sink, series=new_series)
        new_noise.label = self._label
        return new_noise


class MultiNoiseDensity(NoiseDensityBase):
    """Set of noise data series from multiple sources to a single sink"""
    def __init__(self, sources=None, series=None, constituents=None, label=None, **kwargs):
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

        # call parent constructor
        super().__init__(sources=sources, series=series, **kwargs)

        if label is None:
            label = "Incoherent sum"
        self.label = label

        if constituents is not None:
            # check sink agrees with those set in constituents
            if not all([self.sink == spectral_density.sink for spectral_density in constituents]):
                raise Exception("cannot handle noise for functions with different sinks")

    def _set_fallback_plot_options(self):
        """Set plot options in cases where user-specified options are not provided."""
        if "alpha" not in self.plot_options:
            self._set_plot_option("alpha", CONF["plot"]["sum_alpha"])
        if "dash_capstyle" not in self.plot_options:
            self._set_plot_option("dash_capstyle", CONF["plot"]["sum_dash_capstyle"])
        if "linestyle" not in self.plot_options and "ls" not in self.plot_options:
            self._set_plot_option("linestyle", CONF["plot"]["sum_linestyle"])
        if "linewidth" not in self.plot_options and "lw" not in self.plot_options:
            self._set_plot_option("linewidth", CONF["plot"]["sum_linewidth"])
        if "zorder" not in self.plot_options:
            self._set_plot_option("zorder", CONF["plot"]["sum_zorder"])
        # Call parent after setting these options so they don't get overridden.
        super()._set_fallback_plot_options()

    @property
    def noise_names(self):
        if self.constituent_noise is None:
            return "unknown"

        return [spectral_density.noise_name for spectral_density in self.constituent_noise]

    def _format_label(self, *args, suffix=None, ignore_user_label=False, **kwargs):
        if not ignore_user_label and self._label is not None:
            return self._label
        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""
        return f"{self._label}{suffix}"

    def _new_noise_density(self, new_sink, new_series):
        """Create a copy of this noise density with the specified sink and series."""
        new_noise = self.__class__(sources=self.sources, sink=new_sink, series=new_series,
                                   label=self._label)
        return new_noise


class Reference(BaseFunction, metaclass=abc.ABCMeta):
    def __init__(self, frequencies, data, label=None, unit=None, **kwargs):
        self._sink_unit = unit
        super().__init__(series=Series(frequencies, data), **kwargs)
        if label is None:
            label = "Reference"
        self.label = label

    def _format_label(self, *args, suffix=None, ignore_user_label=False, **kwargs):
        if not ignore_user_label and self._label is not None:
            return self._label
        if suffix is not None:
            suffix = " %s" % suffix
        else:
            suffix = ""
        return f"{self._label}{suffix}"

    @property
    def sink_unit(self):
        return self._sink_unit


class ReferenceResponse(Reference, Response):
    pass


class ReferenceNoise(Reference, NoiseDensity):
    pass
