"""Linear circuit simulations"""

import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import logging
from tabulate import tabulate

from .config import ElectronicsConfig
from .components import (Node, Gnd, ImpedanceCoefficient, CurrentCoefficient,
                         VoltageCoefficient)
from .misc import _print_progress, db

LOGGER = logging.getLogger("circuit")
CONF = ElectronicsConfig()

def sparse(*args, **kwargs):
    return lil_matrix(dtype="complex64", *args, **kwargs)

class Circuit(object):
    def __init__(self, input_node=None, input_impedance=0):
        self.components = []
        # ensure ground node is always first node (required for matrix node
        # index methods to correctly function)
        self.nodes = [Gnd()]
        self.input_node = input_node
        self.input_impedance = input_impedance

        # default matrix
        self._matrix = None

    @property
    def input_node(self):
        return self._input_node

    @input_node.setter
    def input_node(self, node):
        self._input_node = node

    @property
    def n_components(self):
        return len(self.components)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def non_gnd_nodes(self):
        for node in self.nodes:
            if node != Gnd():
                yield node

    def add_component(self, component):
        if component in self.components:
            raise ValueError("Component %s already in circuit" % component)

        # add component to end of list
        self.components.append(component)

        # register component's nodes
        for node in component.nodes:
            if node is None:
                raise Exception("Node cannot be none")

            self.add_node(node)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def get_node(self, node_name):
        for node in self.nodes:
            if node.name == node_name:
                return node

    def _construct_matrix(self):
        LOGGER.debug("constructing matrix")

        # create matrix
        self._matrix = sparse((self.dim_size, self.dim_size))

        # dict of methods that accept a frequency parameter
        self._matrix_callables = dict()

        # Ohm's law / op-amp voltage gain equations
        for equation in self.component_equations():
            for coefficient in equation.coefficients:
                # default indices
                row = self._component_matrix_index(equation.component)
                column = self._component_matrix_index(equation.component)

                if isinstance(coefficient, ImpedanceCoefficient):
                    # don't change indices
                    pass
                elif isinstance(coefficient, VoltageCoefficient):
                    # includes extra I[in] column (at index == self.n_components)
                    column = self._voltage_node_matrix_index(coefficient.node)
                else:
                    raise ValueError("Invalid coefficient type")

                # fetch (potentially frequency-dependent) impedance
                if callable(coefficient.value):
                    # copy function
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    # copy value
                    self._matrix[row, column] = coefficient.value

        # first Kirchoff law equations
        for equation in self.node_equations():
            for coefficient in equation.coefficients:
                if not isinstance(coefficient, CurrentCoefficient):
                    raise ValueError("Invalid coefficient type")

                # subtract 1 since 0th row is first component
                row = self._current_node_matrix_index(equation.node)
                column = self._component_matrix_index(coefficient.component)

                if callable(coefficient.value):
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    self._matrix[row, column] = coefficient.value

        # set input voltage to 1
        self._matrix[self._last_index,
                     self._voltage_node_matrix_index(self.input_node)] = 1

        # set input current to 1
        self._matrix[self._current_node_matrix_index(self.input_node),
                     self.n_components] = 1

    def matrix(self, frequency):
        if self._matrix is None:
            # build matrix without frequency dependent values
            self._construct_matrix()

        # copy matrix
        m = self._matrix.copy()

        # substitute frequency into matrix elements
        for coordinates in self._matrix_callables:
            row = coordinates[0]
            column = coordinates[1]

            # call method with current frequency and store in new matrix
            m[row, column] = self._matrix_callables[coordinates](frequency)

        # convert to CSR for efficient solving
        return m.tocsr()

    def solve(self, frequencies, noise_node=None):
        # number of frequencies to calculate
        n_freqs = len(frequencies)

        compute_noise = False
        noise = None

        if noise_node is not None:
            compute_noise = True

            # noise results matrix
            noise = self._results_matrix(n_freqs)

        # signal results matrices
        sig_tfs = self._results_matrix(n_freqs)

        # output vector
        # the last row sets the input voltage to 1
        y = self._results_matrix(1)
        y[self._last_index, 0] = 1

        # create frequency generator with progress bar
        freq_gen = _print_progress(frequencies, n_freqs, update=10)

        # frequency loop
        for freq_index, frequency in enumerate(freq_gen):
            # get matrix for this frequency
            matrix = self.matrix(frequency)

            # solve transfer functions
            sig_tfs[:, freq_index] = spsolve(matrix, y)

            if compute_noise:
                noise[:, freq_index] = self._noise_at_node(noise_node, matrix,
                                                           frequency)

        # create solution
        return Solution(self, frequencies, sig_tfs, noise, noise_node)

    def _noise_at_node(self, node, matrix, frequency):
        e_n = self._results_matrix(1)
        e_n[self._voltage_node_matrix_index(node), 0] = 1

        # set input impedance
        # NOTE: this issues a SparseEfficiencyWarning
        matrix[self._last_index, self.n_components] = self.input_impedance

        # solve, giving transfer function from each component/node to the
        # output (size = nx0)
        y_hat = spsolve(matrix.T, e_n)

        # noise current/voltage input vector (original size = nx1, squeezed
        # size = nx0)
        k = self._noise_input_vector(frequency)

        # noise at output (size = nx1)
        return np.abs(y_hat * k)

    def _noise_input_vector(self, frequency):
        # empty noise current/voltage input vector (size = nx0)
        k = self._results_matrix()

        # fill vector
        for component in self.components:
            # component matrix index
            component_index = self._component_matrix_index(component)

            # component noise potential
            k[component_index] = component.noise_voltage(frequency)

            # FIXME: should really loop over all nodes in circuit separately
            # and add noises in quadrature (in case multiple noise current
            # sources are connected to the same node)

            # component noise currents
            for node, noise_current in component.noise_currents(frequency).items():
                # current node matrix index
                node_index = self._current_node_matrix_index(node)

                # node noise current
                k[node_index] = noise_current

        return k

    @property
    def dim_size(self):
        """Matrix dimension size"""
        return self.n_components + self.n_nodes

    @property
    def _last_index(self):
        return self.dim_size - 1

    def _component_matrix_index(self, component):
        return self.components.index(component)

    def _voltage_node_matrix_index(self, node):
        return self.n_components + self.node_index(node)

    def _current_node_matrix_index(self, node):
        return self.n_components + self.node_index(node) - 1

    def _results_matrix(self, *depth):
        return np.zeros((self.dim_size, *depth), dtype="complex64")

    def component_equations(self):
        return [component.equation() for component in self.components]

    def node_equations(self):
        return [node.equation() for node in self.nodes]

    def node_index(self, node):
        return self.nodes.index(node)

    def coefficients(self):
        for equation in self.equations:
            yield from equation.coefficients

    def print_equations(self, stream=sys.stdout, *args, **kwargs):
        # get matrix
        m = self.matrix(*args, **kwargs)

        for row in range(m.shape[0]):
            # flag to suppress leading sign
            first = True

            for column in range(m.shape[1]):
                element = m[row, column]

                if element == 0:
                    # don't print
                    continue

                if np.sign(element) == -1:
                    print(" - ", end="", file=stream)
                elif not first:
                    print(" + ", end="", file=stream)

                # flag that we're beyond first column
                first = False

                if np.abs(element) != 1:
                    print("%g" % np.abs(element), end="", file=stream)

                print(" %s " % self.formatted_element(column), end="", file=stream)

            if row == m.shape[0] - 1:
                print(" = 1", file=stream)
            else:
                print(" = 0", file=stream)

    def print_matrix(self, stream=sys.stdout, *args, **kwargs):
        # get matrix
        matrix = self.matrix(*args, **kwargs)

        # convert complex values to magnitudes (leaving others, like -1, alone)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.iscomplex(matrix[row, column]):
                    matrix[row, column] = np.abs(matrix[row, column])

        # get matrix as numpy array
        array = matrix.toarray()

        # tabulate data
        table = tabulate(array, self.headers, tablefmt=CONF["format"]["table"])

        # output
        print(table, file=stream)

    @property
    def headers(self):
        return [self.formatted_element(n) for n in range(self.dim_size)]

    def formatted_element(self, index):
        if index < self.n_components:
            return "i[%s]" % self.components[index].name
        elif index == self.n_components:
            # input current
            return "i[in]"
        elif index < self.n_components + 1 + self.n_nodes:
            return "V[%s]" % self.nodes[index - self.n_components]
        else:
            raise ValueError("Invalid element index")

class Solution(object):
    def __init__(self, circuit, frequencies, sig_tfs=None, noise=None,
                 noise_node=None):
        self.circuit = circuit
        self.frequencies = frequencies
        self.sig_tfs = sig_tfs
        self.noise = noise
        self.noise_node = noise_node

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    @property
    def sig_tfs(self):
        return self._sig_tfs

    @sig_tfs.setter
    def sig_tfs(self, sig_tfs):
        if sig_tfs is not None:
            # dimension sanity checks
            if sig_tfs.shape != (self.circuit.dim_size, self.n_frequencies):
                raise ValueError("sig_tfs doesn't fit this solution")

        self._sig_tfs = sig_tfs

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
        if output_nodes is None:
            # all nodes except ground
            output_nodes = list(self.circuit.non_gnd_nodes)
        elif isinstance(output_nodes, Node):
            output_nodes = [output_nodes]

        output_nodes = list(output_nodes)

        # output node indices in sig_tfs
        node_indices = [self._result_node_index(node) for node in output_nodes]

        # transfer function
        tfs = self.sig_tfs[node_indices, :]

        # legend
        node_labels = ["%s -> %s" % (self.circuit.input_node, node)
                       for node in output_nodes]

        # plot title
        if not title:
            title = "%s -> %s"

        # make list with output nodes
        if len(output_nodes) > 1:
            output_node_list = "[%s]" % ", ".join([str(node) for node in output_nodes])
        else:
            output_node_list = str(output_nodes[0])

        formatted_title = title % (self.circuit.input_node,
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
            raise Exception("At least one of total and individual flags must "
                            "be set")

        # default noise contributions
        noise = np.zeros((0, self.n_frequencies))

        # default legend
        labels = []

        if individual:
            # add noise contributions
            noise = np.vstack([noise, self.noise])

            # add noise labels
            labels += self.circuit.headers
        if total:
            # incoherent sum of noise
            sum_noise = np.sqrt(np.sum(np.power(self.noise, 2), axis=0))

            # add sum noise
            noise = np.vstack([noise, sum_noise])

            # add sum label
            labels.append("Sum")

        # plot title
        if not title:
            title = "Noise constributions at %s" % self.noise_node

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

        # plot noise from each source
        for label, source in zip(labels, noise):
            if np.all(source) == 0:
                # skip zero noise
                LOGGER.info("skipping zero noise source %s", label)

                # skip this iteration
                continue

            ax.loglog(frequencies, source, label=label)

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
        plt.show()
