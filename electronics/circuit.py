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
    def __init__(self, input_node=None):
        self.components = []
        self.nodes = []
        self.input_node = input_node

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

    @property
    def dim_size(self):
        """Matrix dimension size"""
        return self.n_components + self.n_nodes

    def _construct_matrix(self):
        LOGGER.debug("constructing matrix")

        # create matrix
        self._matrix = sparse((self.dim_size, self.dim_size))

        # dict of methods that accept a frequency parameter
        self._matrix_callables = dict()

        # Ohm's law / op-amp voltage gain equations
        for component_index, equation in enumerate(self.component_equations()):
            for coefficient in equation.coefficients:
                # default indices
                row = component_index
                column = component_index

                if isinstance(coefficient, ImpedanceCoefficient):
                    # don't change indices
                    pass
                elif isinstance(coefficient, VoltageCoefficient):
                    # includes extra I[in] column (at index == self.n_components)
                    column = self.n_components + self.node_index(coefficient.node)
                else:
                    raise ValueError("Invalid coefficient type")

                if callable(coefficient.value):
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    self._matrix[row, column] = coefficient.value

        # first Kirchoff law equations
        for node_index, equation in enumerate(self.node_equations()):
            for coefficient in equation.coefficients:
                if not isinstance(coefficient, CurrentCoefficient):
                    raise ValueError("Invalid coefficient type")

                # get component index
                component_index = self.component_index(coefficient.component)

                # subtract 1 since 0th row is first component
                row = self.n_components - 1 + node_index
                column = component_index

                if callable(coefficient.value):
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    self._matrix[row, column] = coefficient.value

        # input voltage
        self._matrix[self.n_components + self.n_nodes - 1,
                     self.n_components + self.node_index(self.input_node)] = 1

        # input current
        self._matrix[self.n_components - 1 + self.node_index(self.input_node),
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

    def solve(self, frequencies):
        # number of frequencies to calculate
        n_freqs = len(frequencies)

        # signal and noise transfer function results matrices
        sig_tf = self._results_matrix(n_freqs)
        noise_tf = self._results_matrix(n_freqs)

        # output vector
        # the last row sets the input voltage to 1
        y = sparse((self.dim_size, 1))
        y[-1, 0] = 1

        # create frequency generator with progress bar
        freq_gen = _print_progress(frequencies, n_freqs, update=10)

        # frequency loop
        for index, frequency in enumerate(freq_gen):
            # get matrix for this frequency
            m = self.matrix(frequency)

            # solve
            sig_tf[:, index] = spsolve(m, y)

        # create solution
        return Solution(self, frequencies, sig_tf)

    def _results_matrix(self, depth):
        return np.zeros((self.dim_size, depth), dtype="complex64")

    def component_equations(self):
        return [component.equation() for component in self.components]

    def node_equations(self):
        return [node.equation() for node in self.nodes]

    def component_index(self, component):
        return self.components.index(component)

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
                element = np.abs(m[row, column])

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
        # get matrix as numpy array of absolute values
        m_array = np.abs(self.matrix(*args, **kwargs).toarray())

        # tabulate data
        table = tabulate(m_array, self.headers(),
                         tablefmt=CONF["format"]["table"])

        # output
        print(table, file=stream)

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
    def __init__(self, circuit, frequencies, sig_tf=None, noise_tf=None):
        self.circuit = circuit
        self.frequencies = frequencies
        self.sig_tf = sig_tf
        self.noise_tf = noise_tf

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    @property
    def sig_tf(self):
        return self._sig_tf

    @sig_tf.setter
    def sig_tf(self, sig_tf):
        if sig_tf is not None:
            # dimension sanity checks
            if sig_tf.shape != (self.circuit.dim_size, self.n_frequencies):
                raise ValueError("sig_tf doesn't fit this solution")

        self._sig_tf = sig_tf

    @property
    def noise_tf(self):
        return self._noise_tf

    @noise_tf.setter
    def noise_tf(self, noise_tf):
        if noise_tf is not None:
            # dimension sanity checks
            if noise_tf.shape != (self.circuit.dim_size, self.n_frequencies):
                raise ValueError("sig_tf doesn't fit this solution")

        self._noise_tf = noise_tf

    def _result_node_index(self, node):
        return (self.circuit.n_components
                + self.circuit.node_index(node))

    def plot_sig_tf(self, output_nodes=None, title=None):
        if output_nodes is None:
            # all nodes except ground
            output_nodes = list(self.circuit.non_gnd_nodes)
        elif isinstance(output_nodes, Node):
            output_nodes = [output_nodes]

        # output node indices in sig_tf
        node_indices = [self._result_node_index(node) for node in output_nodes]

        # transfer function
        tfs = self.sig_tf[node_indices, :]

        # legend
        node_labels = ["%s -> %s" % (self.circuit.input_node, node)
                       for node in output_nodes]

        # plot title
        if not title:
            title = CONF["plot"]["default_sig_tf_title"]

        # make list with output nodes
        if len(output_nodes) > 1:
            output_node_list = "[%s]" % ", ".join([str(node) for node in output_nodes])
        else:
            output_node_list = str(output_nodes[0])

        formatted_title = title % (self.circuit.input_node,
                                   output_node_list)

        return self._plot_tfs(self.frequencies, tfs, legend=node_labels,
                             title=formatted_title)

    @staticmethod
    def _plot_tfs(frequencies, tfs, legend=None, legend_loc="best", title=None):
        # create figure
        fig = plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                  float(CONF["plot"]["size_y"])))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        for tf in tfs:
            # plot magnitude
            ax1.semilogx(frequencies, db(np.abs(tf)))

            # plot phase
            ax2.semilogx(frequencies, np.angle(tf) * 180 / np.pi)

        # overall figure title
        fig.suptitle(title)

        # legend
        if legend:
            ax1.legend(legend, loc=legend_loc)

        # set axis properties
        ax2.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax2.set_ylabel("Phase (°)")
        ax1.grid(True)
        ax2.grid(True)

        plt.show()
