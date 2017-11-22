"""Plotting functions for solutions to simulations"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import abc

from .config import ElectronicsConfig
from .data import Series, TransferFunction
from .misc import db

LOGGER = logging.getLogger("solution")
CONF = ElectronicsConfig()

class Solution(object):
    """Represents a solution to the simulated circuit"""

    def __init__(self, circuit, frequencies, tfs=None, noise=None,
                 noise_node=None):
        """Instantiate a new solution

        :param circuit: circuit this solution represents
        :type circuit: :class:`~Circuit`
        :param frequencies: sequence of frequencies this solution contains \
                            results for
        :type frequencies: :class:`~np.ndarray`
        :param tfs: transfer function solutions
        :type tfs: :class:`~np.ndarray`
        :param noise: noise solutions
        :type noise: :class:`~np.ndarray`
        :param noise_node: node ``noise`` represents
        :type noise_node: :class:`~Node`
        """

        self.circuit = circuit
        self.frequencies = frequencies

        # defaults
        self.functions = []

        # process inputs
        self._handle_tfs(tfs)
        self._handle_noise(noise, noise_node)

    def _handle_tfs(self, tfs):
        if tfs is None:
            return

        # dimension sanity checks
        if tfs.shape != (self.circuit.dim_size, self.n_frequencies):
            raise ValueError("tfs doesn't fit this solution")

        # output node indices in tfs
        node_indices = [self._result_node_index(node)
                        for node in self.circuit.non_gnd_nodes]

        # node transfer functions
        node_tfs = tfs[node_indices, :]

        # create functions from each row
        for tf, sink in zip(node_tfs, self.circuit.non_gnd_nodes):
            # source is always input node
            source = self.circuit.input_nodes[0]

            # create series
            series = Series(x=self.frequencies, y=tf)

            # create transfer function
            self.add_function(TransferFunction(source=source, sink=sink,
                                               series=series))

    def _handle_noise(self, noise, sink):
        if noise is None:
            return

        # dimension sanity checks
        if noise.shape != (self.circuit.dim_size, self.n_frequencies):
            raise ValueError("noise doesn't fit this solution")

        # noise sources
        # FIXME: use components/nodes
        sources = self.circuit.column_headers

        # skipped noise sources
        skips = []

        # create functions from each row
        for spectrum, source in zip(noise, sources):
            if np.all(spectrum) == 0:
                # skip zero noise source
                skips.append(source)

                # skip this iteration
                continue

            # FIXME: use isinstance() to set proper noise type

        if len(skips):
            LOGGER.info("skipped zero noise sources %s", ", ".join(skips))

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

    def transfer_functions(self, *args, **kwargs):
        return self._filter_function(TransferFunction, *args, **kwargs)

    def noise_functions(self, *args, **kwargs):
        return self._filter_function(NoiseSpectrum, *args, **kwargs)

    def _result_node_index(self, node):
        return (self.circuit.n_components
                + self.circuit.node_index(node))

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
                   ylabel_mag="Magnitude (dB)", ylabel_phase="Phase (°)"):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for tf in tfs:
            # plot magnitude
            ax1.semilogx(tf.series.x, db(np.abs(tf.series.y)), label=tf.label)

            # plot phase
            ax2.semilogx(tf.series.x, np.angle(tf.series.y) * 180 / np.pi)

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

    def plot_noise(self, total=True, individual=True, title=None):
        if self.noise is None:
            raise Exception("noise was not computed in this solution")

        if not total and not individual:
            raise Exception("at least one of total and individual flags must "
                            "be set")

        # default noise contributions
        noise = np.zeros((0, self.n_frequencies))

        # default legend
        labels = []

        if individual:
            # add noise contributions
            noise = np.vstack([noise, self.noise])

            # add noise labels
            labels += self.circuit.column_headers

        if total:
            # incoherent sum of noise
            sum_noise = np.sqrt(np.sum(np.power(self.noise, 2), axis=0))

            # add sum noise
            noise = np.vstack([noise, sum_noise])

            # add sum label
            labels.append("Sum")

        # plot title
        if not title:
            title = "Noise contributions at %s" % self.noise_node

        return self._plot_noise(self.frequencies, noise, labels=labels,
                                title=title)

    @staticmethod
    def _plot_noise(frequencies, noise, labels, legend=True, legend_loc="best",
                    title=None, xlim=None, ylim=None, xlabel="Frequency (Hz)",
                    ylabel="Noise ()"):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax = fig.gca()

        skips = []
        # plot noise from each source
        for label, source in zip(labels, noise):
            if np.all(source) == 0:
                # skip zero noise
                skips.append(label)

                # skip this iteration
                continue

            ax.loglog(frequencies, source, label=label)

        LOGGER.info("skipping zero noise source %s", ", ".join(skips))

        # overall figure title
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
