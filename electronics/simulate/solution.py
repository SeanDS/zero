"""Plotting functions for solutions to simulations"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import abc

from ..config import ElectronicsConfig
from ..data import Series, TransferFunction, NoiseSpectrum, MultiNoiseSpectrum

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

        return [function.sink for function in self.transfer_functions()]

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
    def has_tfs(self):
        return len(list(self.transfer_functions())) > 0

    @property
    def has_noise(self):
        return len(list(self.noise_functions())) > 0

    def transfer_functions(self, *args, **kwargs):
        return self._filter_function(TransferFunction, *args, **kwargs)

    def noise_functions(self, *args, **kwargs):
        return self._filter_function(NoiseSpectrum, *args, **kwargs)

    def _result_node_index(self, node):
        return (self.circuit.n_components
                + self.circuit.node_index(node))

    def plot(self):
        """Plot all available functions"""

        if self.has_tfs:
            self.plot_tf()

        if self.has_noise:
            self.plot_noise()

        self.show()

    def plot_tf(self, output_nodes=None, title=None):
        if not len(self.transfer_functions()):
            raise Exception("transfer functions were not computed in this solution")

        # work out which output nodes to plot
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

        return self._plot_bode(self.frequencies, tfs, title=title)

    @staticmethod
    def _plot_bode(frequencies, tfs, legend=True, legend_loc="best",
                   title=None, xlim=None, ylim=None, xlabel="Frequency (Hz)",
                   ylabel_mag="Magnitude (dB)",
                   ylabel_phase=r"Phase ($\degree$)", xtick_major_step=20,
                   xtick_minor_step=10, ytick_major_step=30,
                   ytick_minor_step=15):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

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

    def plot_noise(self, total=True, individual=True, title=None):
        if not len(self.noise_functions()):
            raise Exception("noise was not computed in this solution")

        if not total and not individual:
            raise Exception("at least one of total and individual flags must "
                            "be set")

        noise = list(self.noise_functions())

        if total:
            # create combined noise spectrum
            noise.append(MultiNoiseSpectrum(noise))

        return self._plot_noise(self.frequencies, noise, title=title)

    @staticmethod
    def _plot_noise(frequencies, noise, legend=True, legend_loc="best",
                    title=None, xlim=None, ylim=None, xlabel="Frequency (Hz)",
                    ylabel=r"Noise ($\frac{\mathrm{V}}{\sqrt{\mathrm{Hz}}}$)"):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax = fig.gca()

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
        ax.set_ylabel(ylabel)
        ax.grid(True)

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
