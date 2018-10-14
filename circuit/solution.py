"""Plotting functions for solutions to simulations"""

import logging
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .config import CircuitConfig
from .data import TransferFunction, NoiseSpectrum, MultiNoiseSpectrum, SumNoiseSpectrum
from .components import Component, Node, Noise

LOGGER = logging.getLogger(__name__)
CONF = CircuitConfig()

class Solution:
    """Represents a solution to the simulated circuit"""

    # filter flags
    TF_SOURCES_ALL = "all"
    TF_SINKS_ALL = "all"
    NOISE_SOURCES_ALL = "all"
    NOISE_SINKS_ALL = "all"
    NOISE_TYPES_ALL = "all"

    def __init__(self, frequencies):
        """Instantiate a new solution

        :param frequencies: sequence of frequencies this solution contains \
                            results for
        :type frequencies: :class:`~np.ndarray`
        """
        # defaults
        self.functions = []
        self._default_tfs = []
        self._default_noise = []
        self._default_noise_sums = []

        self.frequencies = frequencies

    def add_tf(self, tf, default=False):
        """Add a transfer function to the solution.

        Parameters
        ----------
        tf : :class:`.TransferFunction`
            The transfer function to add.
        default : :class:`bool`, optional
            Whether this transfer function is a default.

        Raises
        ------
        ValueError
            If the specified transfer function is incompatible with this solution.
        """
        # dimension sanity checks
        if not np.all(tf.frequencies == self.frequencies):
            raise ValueError("transfer function '%s' doesn't fit this solution" % tf)

        self._add_function(tf)

        if default:
            self.set_tf_as_default(tf)

    def set_tf_as_default(self, tf):
        """Set the specified transfer function as a default"""
        if tf not in self.tfs:
            raise ValueError("transfer function '%s' is not in the solution" % tf)

        if tf in self._default_tfs:
            raise ValueError("transfer function '%s' is already default" % tf)

        self._default_tfs.append(tf)

    def add_noise(self, spectrum, default=False):
        """Add a noise spectrum to the solution.

        Parameters
        ----------
        spectrum : :class:`.NoiseSpectrum`
            The noise spectrum to add.
        default : :class:`bool`, optional
            Whether this noise spectrum is a default.

        Raises
        ------
        ValueError
            If the specified noise spectrum is incompatible with this solution or not single-source
            or single-sink.
        """
        if not isinstance(spectrum, NoiseSpectrum):
            raise ValueError("noise spectrum '%s' is not single-source and -sink type" % spectrum)

        # dimension sanity checks
        if not np.all(spectrum.frequencies == self.frequencies):
            raise ValueError("noise spectrum '%s' doesn't fit this solution" % spectrum)

        self._add_function(spectrum)

        if default:
            self.set_noise_as_default(spectrum)

    def set_noise_as_default(self, spectrum):
        """Set the specified noise spectrum as a default"""
        if spectrum not in self.noise:
            raise ValueError("noise spectrum '%s' is not in the solution" % spectrum)

        if spectrum in self._default_noise:
            raise ValueError("noise spectrum '%s' is already default" % spectrum)

        self._default_noise.append(spectrum)

    def add_noise_sum(self, noise_sum, default=False):
        """Add a noise sum to the solution.

        Parameters
        ----------
        noise_sum : :class:`.MultiNoiseSpectrum` or :class:`.SumNoiseSpectrum`
            The noise sum to add.
        default : :class:`bool`, optional
            Whether this noise sum is a default.

        Raises
        ------
        ValueError
            If the specified noise sum is incompatible with this solution or not multi-source.
        """
        if not isinstance(noise_sum, (MultiNoiseSpectrum, SumNoiseSpectrum)):
            raise ValueError("noise sum '%s' is not multi-source type" % noise_sum)

        # dimension sanity checks
        if not np.all(noise_sum.frequencies == self.frequencies):
            raise ValueError("noise sum '%s' doesn't fit this solution" % noise_sum)

        self._add_function(noise_sum)

        if default:
            self.set_noise_sum_as_default(noise_sum)

    def set_noise_sum_as_default(self, noise_sum):
        """Set the specified noise sum as a default"""
        if noise_sum not in self.noise_sums:
            raise ValueError("noise sum '%s' is not in the solution" % noise_sum)

        if noise_sum in self._default_noise_sums:
            raise ValueError("noise sum '%s' is already default" % noise_sum)

        self._default_noise_sums.append(noise_sum)

    def _add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    def filter_tfs(self, **kwargs):
        return self._apply_tf_filters(self.tfs, **kwargs)

    def _apply_tf_filters(self, tfs, sources=None, sinks=None):
        filter_sources = []
        filter_sinks = []

        if sources is None:
            sources = self.TF_SOURCES_ALL
        if sinks is None:
            sinks = self.TF_SINKS_ALL

        if sources != self.TF_SOURCES_ALL:
            if isinstance(sources, str):
                sources = [sources]

            for source in sources:
                if isinstance(source, str):
                    source = self.get_tf_source(source)

                if not isinstance(source, (Component, Node)):
                    raise ValueError("signal source '%s' is not a component or node" % source)

                filter_sources.append(source)

            # filter by source
            tfs = [tf for tf in tfs if tf.source in filter_sources]

        if sinks != self.TF_SINKS_ALL:
            if isinstance(sinks, str):
                sinks = [sinks]

            for sink in sinks:
                if isinstance(sink, str):
                    sink = self.get_tf_sink(sink)

                if not isinstance(sink, (Component, Node)):
                    raise ValueError("signal sink '%s' is not a component or node" % sink)

                filter_sinks.append(sink)

            # filter by sink
            tfs = [tf for tf in tfs if tf.sink in filter_sinks]

        return tfs

    def filter_noise(self, **kwargs):
        """Filter for noise spectra.

        This does not include sums.
        """
        return self._apply_noise_filters(self.noise, **kwargs)

    def _filter_default_noise(self, **kwargs):
        """Special filter for default noise spectra.

        This does not include default sums.
        """
        return self._apply_noise_filters(self._default_noise, **kwargs)

    def _apply_noise_filters(self, spectra, sources=None, sinks=None, types=None):
        filter_sources = []
        filter_sinks = []

        if sources is None:
            sources = self.NOISE_SOURCES_ALL
        if sinks is None:
            sinks = self.NOISE_SINKS_ALL
        if types is None:
            types = self.NOISE_TYPES_ALL

        if sources != self.NOISE_SOURCES_ALL:
            if isinstance(sources, str):
                sources = [sources]

            for source in sources:
                if isinstance(source, str):
                    source = self.get_noise_source(source)

                if not isinstance(source, Noise):
                    raise ValueError("noise source '%s' is not a noise source" % source)

                filter_sources.append(source)

            # filter by source
            spectra = [spectrum for spectrum in spectra if spectrum.source in filter_sources]

        if sinks != self.NOISE_SINKS_ALL:
            if isinstance(sinks, str):
                sinks = [sinks]

            for sink in sinks:
                if isinstance(sink, str):
                    sink = self.get_noise_sink(sink)

                if not isinstance(sink, Node):
                    raise ValueError("noise sink '%s' is not a node" % sink)

                filter_sinks.append(sink)

            # filter by sink
            spectra = [spectrum for spectrum in spectra if spectrum.sink in filter_sinks]

        if types != self.NOISE_TYPES_ALL:
            # filter by noise type
            for spectrum in spectra:
                if spectrum.noise_type not in types and spectrum.noise_subtype not in types:
                    # no match
                    spectra.remove(spectrum)

        return spectra

    @property
    def tfs(self):
        return [function for function in self.functions if isinstance(function, TransferFunction)]

    @property
    def noise(self):
        return [spectrum for spectrum in self.functions if isinstance(spectrum, NoiseSpectrum)]

    @property
    def noise_sums(self):
        return [spectrum for spectrum in self.functions
                if isinstance(spectrum, (SumNoiseSpectrum, MultiNoiseSpectrum))]

    @property
    def has_tfs(self):
        return len(self.tfs) > 0

    @property
    def has_noise(self):
        return len(self.noise) > 0 or len(self.noise_sums) > 0

    @property
    def default_functions(self):
        """Default transfer functions and noise spectra"""
        yield from self._default_tfs
        yield from self._default_noise
        yield from self._default_noise_sums

    @property
    def tf_source_nodes(self):
        """Get transfer function input nodes.

        :return: input nodes
        :rtype: Set[:class:`Node`]
        """
        return set([tf.source for tf in self.tfs if isinstance(tf.source, Node)])

    @property
    def tf_source_components(self):
        """Get transfer function input components.

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([tf.source for tf in self.tfs if isinstance(tf.source, Component)])

    @property
    def tf_sources(self):
        return self.tf_source_nodes | self.tf_source_components

    def get_tf_source(self, source_name):
        source_name = source_name.lower()

        for source in self.tf_sources:
            if source_name == source.name.lower():
                return source

        raise ValueError("signal source '%s' not found" % source_name)

    @property
    def tf_sink_nodes(self):
        """Get output nodes in solution

        :return: output nodes
        :rtype: Set[:class:`Node`]
        """
        return set([tf.sink for tf in self.tfs if isinstance(tf.sink, Node)])

    @property
    def tf_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([tf.sink for tf in self.tfs if isinstance(tf.sink, Component)])

    @property
    def tf_sinks(self):
        return self.tf_sink_nodes | self.tf_sink_components

    def get_tf_sink(self, sink_name):
        sink_name = sink_name.lower()

        for sink in self.tf_sinks:
            if sink_name == sink.name.lower():
                return sink

        raise ValueError("signal sink '%s' not found" % sink_name)

    @property
    def noise_sources(self):
        """Get noise sources.

        :return: noise sources
        """
        return [spectrum.source for spectrum in self.noise]

    def get_noise_source(self, source_name):
        source_name = source_name.lower()

        for source in self.noise_sources:
            if source_name == source.label().lower():
                return source

        raise ValueError("noise source '%s' not found" % source_name)

    @property
    def noise_sink_nodes(self):
        """Get noise nodes in solution

        :return: noise nodes
        :rtype: Set[:class:`Node`]
        """
        return set([spectrum.sink for spectrum in self.noise if isinstance(spectrum.sink, Node)])

    @property
    def noise_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([spectrum.sink for spectrum in self.noise
                    if isinstance(spectrum.sink, Component)])

    @property
    def noise_sinks(self):
        return self.noise_sink_nodes | self.noise_sink_components

    def get_noise_sink(self, sink_name):
        sink_name = sink_name.lower()

        for sink in self.noise_sinks:
            if sink_name == sink.name.lower():
                return sink

        raise ValueError("noise sink '%s' not found" % sink_name)

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    def plot(self, tf_figure=None, noise_figure=None):
        """Plot all functions contained in this solution using default settings"""
        if self.has_tfs:
            self.plot_tfs(figure=tf_figure)

        if self.has_noise:
            self.plot_noise(figure=noise_figure)

    def plot_tfs(self, figure=None, sources=None, sinks=None, **kwargs):
        """Plot transfer functions.

        Note: if only one of "sources" or "sinks" is specified, the other defaults to "all" as per
        the behaviour of :meth:`.filter_tfs`.
        """
        if sources is None and sinks is None:
            tfs = self._default_tfs
        else:
            tfs = self.filter_tfs(sources=sources, sinks=sinks)

        if not tfs:
            raise NoDataException("no transfer functions found")

        # draw plot
        figure = self._plot_bode(tfs, figure=figure, **kwargs)
        LOGGER.info("tf(s) plotted on %s", figure.canvas.get_window_title())

        return figure

    def plot_noise(self, figure=None, sources=None, sinks=None, types=None, show_sums=True,
                   title=None):
        """Plot noise.

        Note: if only some of "sources", "sinks" and "types" are specified, the others default to
        "all" as per the behaviour of :meth:`.filter_noise`.
        """
        if sources is None and sinks is None and types is None:
            # filter against sum flag
            noise = self._filter_default_noise()

            if show_sums:
                noise.extend(self._default_noise_sums)
        else:
            noise = self.filter_noise(sources=sources, sinks=sinks, types=types)

            if show_sums:
                noise.extend(self.noise_sums)

        if not noise:
            raise NoDataException("no noise spectra found from specified sources and/or sum(s)")

        figure = self._plot_spectrum(noise, figure=figure, title=title)
        LOGGER.info("noise plotted on %s", figure.canvas.get_window_title())

        return figure

    def figure(self):
        return plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                   float(CONF["plot"]["size_y"])))

    def bode_figure(self):
        figure = self.figure()
        ax1 = figure.add_subplot(211)
        _ = figure.add_subplot(212, sharex=ax1)

        return figure

    def noise_figure(self):
        figure = self.figure()
        _ = figure.add_subplot(111)

        return figure

    def _plot_bode(self, tfs, figure=None, legend=True, legend_loc="best",
                   title=None, xlim=None, ylim=None, xlabel=r"$\bf{Frequency}$ (Hz)",
                   ylabel_mag=r"$\bf{Magnitude}$ (dB)", ylabel_phase=r"$\bf{Phase}$ ($\degree$)",
                   mag_tick_major_step=20, mag_tick_minor_step=10, phase_tick_major_step=30,
                   phase_tick_minor_step=15):
        if figure is None:
            # create figure
            figure = self.bode_figure()

        if len(figure.axes) != 2:
            raise ValueError("specified figure must contain two axes")

        ax1, ax2 = figure.axes

        for tf in tfs:
            tf.draw(ax1, ax2)

        # overall figure title
        if title:
            figure.suptitle(title)

        # legend
        if legend:
            ax1.legend(loc=legend_loc)

        # limits
        if xlim:
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)
        if ylim:
            ax1.set_ylim(ylim)
            ax2.set_ylim(ylim)

        # set other axis properties
        ax2.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel_mag)
        ax2.set_ylabel(ylabel_phase)
        ax1.grid(True)
        ax2.grid(True)

        # magnitude and phase tick locators
        ax1.yaxis.set_major_locator(MultipleLocator(base=mag_tick_major_step))
        ax1.yaxis.set_minor_locator(MultipleLocator(base=mag_tick_minor_step))
        ax2.yaxis.set_major_locator(MultipleLocator(base=phase_tick_major_step))
        ax2.yaxis.set_minor_locator(MultipleLocator(base=phase_tick_minor_step))

        return figure

    def _plot_spectrum(self, noise, figure=None, legend=True, legend_loc="best", title=None,
                       xlim=None, ylim=None, xlabel=r"$\bf{Frequency}$ (Hz)",
                       ylabel=r"$\bf{Noise}$ ($\frac{\mathrm{V}}{\sqrt{\mathrm{Hz}}}$)"):
        if figure is None:
            # create figure
            figure = self.noise_figure()

        if len(figure.axes) != 1:
            raise ValueError("specified figure must contain one axis")

        ax = figure.axes[0]

        for spectrum in noise:
            spectrum.draw(ax)

        # overall figure title
        if title:
            figure.suptitle(title)

        # legend
        if legend:
            ax.legend(loc=legend_loc)

        # limits
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # set other axis properties
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        return figure

    def plot_style_context(self, *args, **kwargs):
        """Plot style context manager, used to override the default style"""
        return plt.rc_context(*args, **kwargs)

    @staticmethod
    def show():
        """Show plots"""
        plt.tight_layout()
        plt.show()

    def equivalent_defaults(self, other):
        """Checks if the specified other solution has equivalent, identical displayed plots"""
        # check frequencies match
        if np.all(self.frequencies != other.frequencies):
            return False

        our_functions = list(self.default_functions)
        their_functions = list(other.default_functions)

        LOGGER.debug("comparing %i / %i functions", len(our_functions), len(their_functions))
        LOGGER.debug("this solution's functions:")
        for function in our_functions:
            LOGGER.debug("%s (%s)", function, function.__class__)
        LOGGER.debug("other solution's functions:")
        for function in their_functions:
            LOGGER.debug("%s (%s)", function, function.__class__)

        # check functions match
        for function in our_functions:
            if function in their_functions:
                # match
                their_functions.remove(function)

        if their_functions:
            LOGGER.info("non-matches: %s", ", ".join([str(n) for n in their_functions]))
            # some functions didn't match
            return False

        return True

class NoDataException(Exception):
    pass
