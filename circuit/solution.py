"""Plotting functions for solutions to simulations"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import abc

from .config import CircuitConfig
from .data import (TransferFunction, VoltageVoltageTF, VoltageCurrentTF,
                   CurrentCurrentTF, CurrentVoltageTF, NoiseSpectrum,
                   MultiNoiseSpectrum, Series)
from .components import Component, Node, Noise

LOGGER = logging.getLogger("solution")
CONF = CircuitConfig()

class Solution(object):
    """Represents a solution to the simulated circuit"""

    def __init__(self, circuit, frequencies):
        """Instantiate a new solution

        :param circuit: circuit this solution represents
        :type circuit: :class:`~Circuit`
        :param frequencies: sequence of frequencies this solution contains \
                            results for
        :type frequencies: :class:`~np.ndarray`
        """

        self.circuit = circuit
        self.frequencies = frequencies

        # defaults
        self.functions = []

    def add_tf(self, tf):
        """Add a transfer function to the solution

        :param tf: transfer function
        :type tf: :class:`~TransferFunction`
        """

        # dimension sanity checks
        if not np.all(tf.frequencies == self.frequencies):
            raise ValueError("tf doesn't fit this solution")

        self._add_function(tf)

    def add_noise(self, noise_spectrum):
        """Add noise spectrum to the solution

        :param noise_spectrum: noise spectrum
        :type noise_spectrum: :class:`~NoiseSpectrum`
        """

        # dimension sanity checks
        if not np.all(noise_spectrum.frequencies == self.frequencies):
            raise ValueError("noise spectrum doesn't fit this solution")

        self._add_function(noise_spectrum)

    def _add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    @property
    def tfs(self):
        return [function for function in self.functions if isinstance(function, TransferFunction)]

    @property
    def noise(self):
        return [function for function in self.functions if isinstance(function, NoiseSpectrum)]

    @property
    def has_tfs(self):
        return len(self.tfs) > 0

    @property
    def has_noise(self):
        return len(self.noise) > 0

    @property
    def tf_source_nodes(self):
        """Get transfer function input nodes.

        :return: input nodes
        :rtype: Set[:class:`Node`]
        """
        return set([function.source for function in self.tfs if isinstance(function.source, Node)])

    @property
    def tf_source_components(self):
        """Get transfer function input components.

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([function.source for function in self.tfs if isinstance(function.source, Component)])

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
        return set([function.sink for function in self.tfs if isinstance(function.sink, Node)])

    @property
    def tf_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([function.sink for function in self.tfs if isinstance(function.sink, Component)])

    @property
    def tf_sinks(self):
        return self.tf_sink_nodes | self.tf_sink_components

    def get_tf_sink(self, sink_name):
        sink_name = sink_name.lower()

        print([t.sink for t in self.tfs])

        for sink in self.tf_sinks:
            if sink_name == sink.name.lower():
                return sink
        
        raise ValueError("signal sink '%s' not found" % sink_name)

    @property
    def noise_sources(self):
        """Get noise sources.

        :return: noise sources
        """
        return [function.source for function in self.noise]

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
        return set([function.sink for function in self.noise if isinstance(function.sink, Node)])

    @property
    def noise_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        return set([function.sink for function in self.noise if isinstance(function.sink, Component)])

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

    def _filter_function(self, _class, sources=None, sinks=None):
        functions = [function for function in self.functions
                     if isinstance(function, _class)]

        # filter by source
        if sources is not None:
            functions = [function for function in functions
                         if function.source in sources]

        # filter by sink
        if sinks is not None:
            functions = [function for function in functions
                         if function.sink in sinks]

        return functions

    def filter_tfs(self, sources=None, sinks=None):
        source_elements = []
        sink_elements = []

        if not sources:
            # all sources
            source_elements = self.tf_sources
        else:
            if isinstance(sources, str):
                # parse source name
                source_elements.append(self.get_tf_source(sources))
            else:
                # assume list
                for source in sources:
                    if isinstance(source, str):
                        # parse source name
                        source_elements.append(self.get_tf_source(source))
                    else:
                        if not isinstance(source, (Component, Node)):
                            raise ValueError("signal source '%s' is not a component or node" % source)
                        
                        source_elements.append(source)

        if not sinks:
            # all sinks
            sink_elements = self.tf_sinks
        else:
            if isinstance(sinks, str):
                # parse source name
                sink_elements.append(self.get_tf_sink(sinks))
            else:
                # assume list
                for sink in sinks:
                    if isinstance(sink, str):
                        # parse source name
                        sink_elements.append(self.get_tf_sink(sink))
                    else:
                        if not isinstance(sink, (Component, Node)):
                            raise ValueError("signal sink '%s' is not a component or node" % sink)
                        
                        sink_elements.append(sink)

        # filter transfer functions
        return self._filter_function(TransferFunction, sources=source_elements, sinks=sink_elements)

    def filter_noise(self, sources=None, sinks=None):
        source_elements = []
        sink_elements = []

        if not sources:
            # all sources
            source_elements = self.noise_sources
        else:
            if isinstance(sources, str):
                # parse source name
                source_elements.append(self.get_noise_source(sources))
            else:
                # assume list
                for source in sources:
                    if isinstance(source, str):
                        # parse source name
                        source_elements.append(self.get_noise_source(source))
                    else:
                        if not isinstance(source, Noise):
                            raise ValueError("noise source '%s' is not a noise source" % source)
                        
                        source_elements.append(source)

        if not sinks:
            # all sinks
            sink_elements = self.noise_sinks
        else:
            if isinstance(sinks, str):
                # parse source name
                sink_elements.append(self.get_noise_sink(sinks))
            else:
                # assume list
                for sink in sinks:
                    if isinstance(sink, str):
                        # parse source name
                        sink_elements.append(self.get_noise_sink(sink))
                    else:
                        if not isinstance(sink, Node):
                            raise ValueError("noise sink '%s' is not a node" % sink)
                        
                        sink_elements.append(sink)

        # filter noise spectra
        return self._filter_function(NoiseSpectrum, sources=source_elements, sinks=sink_elements)

    def _result_node_index(self, node):
        return self.circuit.n_components + self.circuit.node_index(node)

    def plot_tfs(self, figure=None, sources=None, sinks=None, **kwargs):
        if figure is None:
            figure = self.bode_figure()

        # transfer functions between the sources and sinks
        tfs = self.filter_tfs(sources=sources, sinks=sinks)

        if not tfs:
            raise NoDataException("no transfer functions found between specified sources and sinks")

        # draw plot
        figure = self._plot_bode(self.frequencies, tfs, figure=figure, **kwargs)
        LOGGER.info("tf(s) plotted on %s", figure.canvas.get_window_title())

        return figure

    def figure(self):
        return plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                   float(CONF["plot"]["size_y"])))

    def bode_figure(self):
        figure = self.figure()
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212, sharex=ax1)

        return figure

    def noise_figure(self):
        figure = self.figure()
        ax1 = figure.add_subplot(111)

        return figure

    def _plot_bode(self, frequencies, tfs, figure=None, legend=True, legend_loc="best",
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

    def plot_noise(self, figure=None, sources=None, sinks=None, total=True, individual=True,
                   title=None):
        if not total and not individual:
            raise Exception("at least one of total and individual flags must "
                            "be set")
        
        # get noise
        noise = self.filter_noise(sources=sources, sinks=sinks)

        if not noise:
            raise NoDataException("no noise spectra found between specified sources and sinks")

        if total:
            # create combined noise spectrum
            noise.append(MultiNoiseSpectrum(noise))

        figure = self._plot_noise(self.frequencies, noise, figure=figure, title=title)
        LOGGER.info("noise plotted on %s", figure.canvas.get_window_title())

        return figure

    def _plot_noise(self, frequencies, noise, figure=None, legend=True,
                    legend_loc="best", title=None, xlim=None, ylim=None,
                    xlabel=r"$\bf{Frequency}$ (Hz)",
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

    @staticmethod
    def show():
        """Show plots"""
        plt.show()

    def __eq__(self, other):
        # check frequencies match
        if np.all(self.frequencies != other.frequencies):
            return False

        # check circuits match
        if self.circuit != other.circuit:
            return False

        LOGGER.info("comparing %i / %i functions" % (len(self.functions),
                                                     len(other.functions)))
        LOGGER.info("first solution's functions:")
        for f in self.functions:
            LOGGER.info("%s (%s)", f, f.__class__)
        LOGGER.info("second solution's functions:")
        for f in other.functions:
            LOGGER.info("%s (%s)", f, f.__class__)

        # check functions match
        other_tfs = other.tfs
        other_noise = other.noise

        for tf in self.tfs:
            # with builtin list object, "in" uses __eq__
            if tf in other_tfs:
                # we have a match
                other_tfs.remove(tf)

        for noise in self.noise:
            # with builtin list object, "in" uses __eq__
            if noise in other_noise:
                # we have a match
                other_noise.remove(noise)

        if len(other_tfs) > 0 or len(other_noise) > 0:
            LOGGER.info("non-matches: %s",
                        ", ".join([str(n) for n in other_tfs + other_noise]))
            # some functions didn't match
            return False

        return True

class NoDataException(Exception):
    pass