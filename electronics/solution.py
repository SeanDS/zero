"""Plotting functions for solutions to simulations"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import abc

from .config import ElectronicsConfig
from .misc import db

LOGGER = logging.getLogger("solution")
CONF = ElectronicsConfig()

class Solution(object):
    """Represents a solution to the simulated circuit"""

    def __init__(self, circuit, frequencies, tfs=None, noise=None, noise_node=None):
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
        self.tfs = tfs
        self.noise = noise
        self.noise_node = noise_node

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    @property
    def tfs(self):
        return self._tfs

    @tfs.setter
    def tfs(self, tfs):
        if tfs is not None:
            # dimension sanity checks
            if tfs.shape != (self.circuit.dim_size, self.n_frequencies):
                raise ValueError("tfs doesn't fit this solution")

        self._tfs = tfs

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, noise):
        if noise is not None:
            # dimension sanity checks
            if noise.shape != (self.circuit.dim_size, self.n_frequencies):
                raise ValueError("noise doesn't fit this solution")

        self._noise = noise

    def _result_node_index(self, node):
        return (self.circuit.n_components
                + self.circuit.node_index(node))

    def plot_tf(self, output_nodes=None, title=None):
        if self.tfs is None:
            raise Exception("transfer functions were not computed in this solution")

        # work out which output nodes to plot
        if output_nodes is not None:
            # use output node specified in this call
            output_nodes = list(output_nodes)

            # warn user if node is different from default
            if set(output_nodes) != set(self.circuit.default_output_nodes):
                # warn user that nodes differ
                LOGGER.warning("specified output nodes (%s) are not the same as"
                               " circuit's defaults (%s)",
                               ", ".join([str(node) for node in output_nodes]),
                               ", ".join([str(node) for node
                                          in self.circuit.default_output_nodes]))
        else:
            if len(self.circuit.default_output_nodes):
                # use default output nodes
                output_nodes = self.circuit.default_output_nodes
                LOGGER.info("using default output nodes: %s",
                            ", ".join([str(node) for node in output_nodes]))
            else:
                # plot all output nodes
                output_nodes = list(self.circuit.non_gnd_nodes)
                LOGGER.info("plotting all output nodes")

        # output node indices in tfs
        node_indices = [self._result_node_index(node) for node in output_nodes]

        # transfer function
        tfs = self.tfs[node_indices, :]

        # legend
        # FIXME: support floating inputs
        node_labels = ["%s -> %s" % (self.circuit.input_nodes[0], node)
                       for node in output_nodes]

        # plot title
        if not title:
            title = "%s -> %s"

        # make list with output nodes
        if len(output_nodes) > 1:
            output_node_list = "[%s]" % ", ".join([str(node) for node in output_nodes])
        else:
            output_node_list = str(output_nodes[0])

        # FIXME: support floating inputs
        formatted_title = title % (self.circuit.input_nodes[0],
                                   output_node_list)

        return self._plot_bode(self.frequencies, tfs, labels=node_labels,
                               title=formatted_title)

    @staticmethod
    def _plot_bode(frequencies, tfs, labels, legend=True, legend_loc="best",
                   title=None, xlim=None, ylim=None, xlabel="Frequency (Hz)",
                   ylabel_mag="Magnitude (dB)", ylabel_phase="Phase (°)"):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for label, tf in zip(labels, tfs):
            # plot magnitude
            ax1.semilogx(frequencies, db(np.abs(tf)), label=label)

            # plot phase
            ax2.semilogx(frequencies, np.angle(tf) * 180 / np.pi)

        # overall figure title
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
