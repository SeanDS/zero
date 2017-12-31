"""Plotting functions for solutions to simulations"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import abc

from ..config import ElectronicsConfig
from ..data import (Series, CurrentTransferFunction, VoltageTransferFunction,
                    TransferFunction, NoiseSpectrum, MultiNoiseSpectrum)

LOGGER = logging.getLogger("solution")
CONF = ElectronicsConfig()

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

        self.add_function(tf)

    def add_noise(self, noise_spectrum):
        """Add noise spectrum to the solution

        :param noise_spectrum: noise spectrum
        :type noise_spectrum: :class:`~NoiseSpectrum`
        """

        # dimension sanity checks
        if not np.all(noise_spectrum.frequencies == self.frequencies):
            raise ValueError("noise spectrum doesn't fit this solution")

        self.add_function(noise_spectrum)

    def add_function(self, function):
        if function in self.functions:
            raise ValueError("duplicate function")

        self.functions.append(function)

    @property
    def output_nodes(self):
        """Get output nodes in solution

        :return: output nodes
        :rtype: Sequence[:class:`Node`]
        """

        return [function.sink for function in self.voltage_transfer_functions()]

    @property
    def output_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """

        return [function.sink for function in self.current_transfer_functions()]

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    def _filter_function(self, _class, sources=[], sinks=[]):
        functions = [function for function in self.functions
                     if isinstance(function, _class)]

        # filter by source
        if len(sources):
            functions = [function for function in functions
                         if function.source in sources]

        # filter by sink
        if len(sinks):
            functions = [function for function in functions
                         if function.sink in sinks]

        return functions

    @property
    def has_voltage_tfs(self):
        return len(list(self.voltage_transfer_functions())) > 0

    @property
    def has_current_tfs(self):
        return len(list(self.current_transfer_functions())) > 0

    @property
    def has_tfs(self):
        return self.has_voltage_tfs or self.has_current_tfs

    @property
    def has_noise(self):
        return len(list(self.noise_functions())) > 0

    def transfer_functions(self, *args, **kwargs):
        return self._filter_function(TransferFunction, *args, **kwargs)

    def current_transfer_functions(self, *args, **kwargs):
        return self._filter_function(CurrentTransferFunction, *args, **kwargs)

    def voltage_transfer_functions(self, *args, **kwargs):
        return self._filter_function(VoltageTransferFunction, *args, **kwargs)

    def noise_functions(self, *args, **kwargs):
        return self._filter_function(NoiseSpectrum, *args, **kwargs)

    def _result_node_index(self, node):
        return (self.circuit.n_components
                + self.circuit.node_index(node))

    def plot(self):
        """Plot all available functions"""

        if self.has_tfs:
            self.plot_tfs()
        if self.has_noise:
            self.plot_noise()

        self.show()

    def plot_tfs(self, output_components=None, output_nodes=None, *args,
                 **kwargs):
        if not self.has_tfs:
            raise Exception("transfer functions were not computed in this "
                            "solution")

        figure = self.bode_figure()

        if self.has_voltage_tfs:
            self.plot_voltage_tfs(figure=figure, output_nodes=output_nodes,
                                  *args, **kwargs)
        if self.has_current_tfs:
            self.plot_current_tfs(figure=figure,
                                  output_components=output_components,
                                  *args, **kwargs)

    def plot_voltage_tfs(self, figure=None, output_nodes=None, title=None):
        if not self.has_voltage_tfs:
            raise Exception("voltage transfer functions were not computed in "
                            "this solution")

        # work out which sinks to plot
        if output_nodes is not None:
            # use output node specified in this call
            output_nodes = list(output_nodes)

            if not set(output_nodes).issubset(set(self.output_nodes)):
                raise ValueError("not all specified output nodes were computed "
                                 "in solution")
        else:
            # plot all output nodes
            output_nodes = self.output_nodes

        # get transfer functions to specified output nodes
        tfs = self.transfer_functions(sinks=output_nodes)

        figure = self._plot_bode(self.frequencies, tfs, figure=figure,
                                 title=title)
        LOGGER.info("voltage tf(s) plotted on %s",
                    figure.canvas.get_window_title())

        return figure

    def plot_current_tfs(self, figure=None, output_components=None, title=None):
        if not len(self.current_transfer_functions()):
            raise Exception("current transfer functions were not computed in "
                            "this solution")

        # work out which sinks to plot
        if output_components is not None:
            # use output components specified in this call
            output_components = list(output_components)

            if not set(output_components).issubset(set(self.output_components)):
                raise ValueError("not all specified output components were "
                                 "computed in solution")
        else:
            # plot all output components
            output_components = self.output_components

        # get transfer functions to specified output components
        tfs = self.transfer_functions(sinks=output_components)

        figure = self._plot_bode(self.frequencies, tfs, figure=figure,
                                 title=title)
        LOGGER.info("current tf(s) plotted on %s",
                    figure.canvas.get_window_title())

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

    def _plot_bode(self, frequencies, tfs, figure=None, legend=True,
                   legend_loc="best", title=None, xlim=None, ylim=None,
                   xlabel=r"$\bf{Frequency}$ (Hz)",
                   ylabel_mag=r"$\bf{Magnitude}$ (dB)",
                   ylabel_phase=r"$\bf{Phase}$ ($\degree$)", xtick_major_step=20,
                   xtick_minor_step=10, ytick_major_step=30,
                   ytick_minor_step=15):
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
            fig.suptitle(title)

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
        ax1.yaxis.set_major_locator(MultipleLocator(base=xtick_major_step))
        ax1.yaxis.set_minor_locator(MultipleLocator(base=xtick_minor_step))
        ax2.yaxis.set_major_locator(MultipleLocator(base=ytick_major_step))
        ax2.yaxis.set_minor_locator(MultipleLocator(base=ytick_minor_step))

        return figure

    def plot_noise(self, figure=None, total=True, individual=True, title=None):
        if not len(self.noise_functions()):
            raise Exception("noise was not computed in this solution")

        if not total and not individual:
            raise Exception("at least one of total and individual flags must "
                            "be set")

        noise = list(self.noise_functions())

        if total:
            # create combined noise spectrum
            noise.append(MultiNoiseSpectrum(noise))

        figure = self._plot_noise(self.frequencies, noise, figure=figure,
                                  title=title)
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
            fig.suptitle(title)

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

        print("%i / %i functions: " % (len(self.functions), len(other.functions)))
        for f in self.functions:
            print(f, f.__class__)
        print()
        for f in other.functions:
            print(f, f.__class__)

        # check functions match
        other_tfs = other.transfer_functions()
        other_noise = other.noise_functions()

        for tf in self.transfer_functions():
            # with builtin list object, "in" uses __eq__
            if tf in other_tfs:
                # we have a match
                other_tfs.remove(tf)

        for noise in self.noise_functions():
            # with builtin list object, "in" uses __eq__
            if noise in other_noise:
                # we have a match
                other_noise.remove(noise)

        if len(other_tfs) > 0 or len(other_noise) > 0:
            print([str(n) for n in other_noise])
            # some functions didn't match
            return False

        return True
