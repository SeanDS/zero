"""Linear circuit simulations"""

import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import logging
from tabulate import tabulate

from .config import ElectronicsConfig
from .components import (Component, Input, Node, Gnd, ImpedanceCoefficient,
                         CurrentCoefficient, VoltageCoefficient)
from .solution import Solution
from .misc import _print_progress

LOGGER = logging.getLogger("circuit")
CONF = ElectronicsConfig()

def sparse(*args, **kwargs):
    """Create new complex-valued sparse matrix

    :return: sparse matrix
    :rtype: :class:`~lil_matrix`
    """

    # complex64 gives real and imaginary parts each represented as 32-bit floats
    # with 8 bits exponent and 23 bits mantissa, giving between 6 and 7 digits
    # of precision; good enough for most purposes
    return lil_matrix(dtype="complex64", *args, **kwargs)

class Circuit(object):
    """Represents an electronic circuit containing linear components"""

    def __init__(self):
        """Instantiate a new circuit

        A circuit can contain linear components like resistors, capacitors,
        inductors and op-amps.

        Default input, output and noise nodes and impedances can be set with
        ``defaults``
        """

        # default circuit options; can be overridden by external code (e.g.
        # LISO file parser)
        self.defaults = {
            "input_nodes": [],
            "output_nodes": [],
            "noise_node": None,
            "input_impedance": 0
        }

        # solver parameters
        self.input_nodes = None
        self.noise_node = None
        self.input_impedance = None
        self.components = []
        self.nodes = []
        self._matrix = None

    @property
    def non_gnd_nodes(self):
        """Get nodes in circuit, excluding ground

        :return: non-ground nodes
        :rtype: Generator[:class:`~Node`]
        """

        for node in self.nodes:
            if node != Gnd():
                yield node

    @property
    def n_components(self):
        """Get number of components in circuit

        :return: number of components
        :rtype: int
        """

        return len(self.components)

    @property
    def default_input_nodes(self):
        """Get default circuit input nodes

        :return: default input nodes
        :rtype: Sequence[:class:`~Node`]
        """

        return self.defaults["input_nodes"]

    @property
    def default_output_nodes(self):
        """Get default circuit output nodes

        :return: default output nodes
        :rtype: Sequence[:class:`~Node`]
        """

        return self.defaults["output_nodes"]

    @property
    def default_noise_node(self):
        """Get default circuit noise node

        :return: default noise node
        :rtype: :class:`~Node`
        """

        return self.defaults["noise_node"]

    @property
    def default_input_impedance(self):
        """Get default circuit input impedance

        :return: default input impedance
        :rtype: float
        """

        return self.defaults["input_impedance"]

    @property
    def n_nodes(self):
        """Get number of nodes in circuit

        :return: number of nodes
        :rtype: int
        """

        return len(self.nodes)

    def add_component(self, component):
        """Add component to circuit

        :param component: component to add
        :type component: :class:`~Component`
        :raises ValueError: if component is already in the circuit
        """

        if component in self.components:
            raise ValueError("Component %s already in circuit" % component)

        # add component to end of list
        self.components.append(component)

        # register component's nodes
        for node in component.nodes:
            self.add_node(node)

        # reset matrix
        self._matrix = None

    def add_node(self, node):
        """Add node to circuit

        :param node: node to add
        :type node: :class:`~Node`
        :raises Exception: if one of the component's nodes is unspecified
        """

        if node is None:
            raise Exception("Node cannot be none")

        if node not in self.nodes:
            self.nodes.append(node)

    def get_component(self, component_name):
        """Get circuit component by name

        :param component_name: name of component to fetch
        :type component_name: str
        :return: component
        :rtype: :class:`~Component`
        :raises ValueError: if component not found
        """

        for component in self.components:
            if component.name == component_name:
                return component

        raise ValueError("component not found")

    def get_node(self, node_name):
        """Get circuit node by name

        :param node_name: name of node to fetch
        :type node_name: str
        :return: node
        :rtype: :class:`~Node`
        :raises ValueError: if node not found
        """

        for node in self.nodes:
            if node.name == node_name:
                return node

        raise ValueError("node not found")

    def _construct_matrix(self):
        """Construct matrix representing the circuit

        This constructs a sparse matrix containing the voltage and current
        equations for each component and the (optional) input and output nodes.
        The matrix is stored internally in the object so that it can be reused
        as long as the circuit is not changed. Frequency dependent impedances
        are stored as callables so that their values can be quickly calculated
        for a particular frequency during solving.
        """

        LOGGER.debug("constructing matrix")

        # create matrix
        self._matrix = sparse((self.dim_size, self.dim_size))

        # dict of methods that accept a frequency parameter
        self._matrix_callables = dict()

        # Kirchoff's voltage law / op-amp voltage gain equations
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

        # Kirchoff's current law
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

    def matrix(self, frequency):
        """Calculate and return circuit matrix for a given frequency

        Matrix is returned in compressed sparse row (CSR) format for easy
        row access. Further structural modification of the returned matrix is
        inefficient. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html.

        :param frequency: frequency at which to calculate circuit impedances
        :type frequency: float or Numpy scalar
        :return: circuit matrix
        :rtype: :class:`scipy.sparse.spmatrix`
        :raises Exception: if ``set_input`` is True but no inputs are specified
        """

        # generate base matrix if necessary
        if self._matrix is None:
            # build matrix without frequency dependent values
            self._construct_matrix()

        # copy matrix
        matrix = self._matrix.copy()

        # substitute frequency into matrix elements
        for coordinates in self._matrix_callables:
            # extract row and column from tuple
            row, column = coordinates

            # call method with current frequency and store in new matrix
            matrix[row, column] = self._matrix_callables[coordinates](frequency)

        # convert to CSR for efficient solving
        return matrix.tocsr()

    def solve(self, frequencies, input_nodes=[], input_impedance=0,
              noise_node=None, print_progress=True, progress_stream=sys.stdout):
        """Solve matrix for a given input and/or output

        If the input node(s) is/are specified, transfer functions are calculated
        from it to all other nodes in the circuit. If the noise node is
        specified, the noise from all nodes in the circuit projected to the
        noise node is calculated.

        The input and noise nodes can be specified directly, or left as
        defaults. If no default value has previously been set for the circuit
        (for instance from a circuit definition file), then the relevant
        transfer function or noise calculation is not performed. If the input
        node(s) is/are specified, the specified input impedance is also used.
        The circuit's default input impedance is only used if the circuit's
        default input nodes are also being used.

        :param frequencies: sequence of frequencies to solve circuit for
        :type frequencies: Sequence[Numpy scalar or float]
        :param input_nodes: (optional) input nodes to calculate transfer \
                            functions from
        :type input_nodes: Sequence[:class:`~Node`]
        :param input_impedance: (optional) input impedance to assume
        :type input_impedance: float
        :param noise_node: (optional) node to project noise to
        :type noise_node: :class:`~Node`
        :param print_progress: whether to print solve progress to stream
        :type print_progress: bool
        :param progress_stream: stream to print progress to
        :type progress_stream: :class:`io.IOBase`
        :raises Exception: if neither an input nor noise node is specified
        """

        # number of frequencies to calculate
        n_freqs = len(frequencies)

        input_nodes = list(input_nodes)
        input_impedance = float(input_impedance)

        # default values
        compute_tfs = False
        compute_noise = False
        tfs = None
        noise = None

        # work out which input node to use, if any
        if len(input_nodes):
            # use input nodes and impedance specified in this call
            self.input_nodes = input_nodes
            self.input_impedance = input_impedance
            compute_tfs = True

            # warn user if node is different from default
            if set(self.input_nodes) != set(self.default_input_nodes):
                # warn user that nodes differ
                LOGGER.warning("specified input nodes (%s) are not the same as "
                               "circuit's defaults (%s)",
                               ", ".join([str(node) for node in self.input_nodes]),
                               ", ".join([str(node) for node
                                          in self.default_input_nodes]))
        else:
            if len(self.default_input_nodes):
                # use default input node
                self.input_nodes = self.default_input_nodes
                self.input_impedance = self.default_input_impedance
                LOGGER.info("using default input nodes: %s",
                            ", ".join([str(node) for node in self.input_nodes]))

                compute_tfs = True

        # create input component
        input_component = Input()

        # set nodes
        if len(self.input_nodes) == 2:
            input_component.node1 = self.input_nodes[1]
        else:
            input_component.node1 = Gnd()

        input_component.node2 = self.input_nodes[0]

        # add input
        self.add_component(input_component)

        # work out which noise node to use, if any
        if noise_node is not None:
            # use noise node specified in this call
            self.noise_node = noise_node
            compute_noise = True

            # warn user if node is different from default
            if (self.default_noise_node is not None
                and noise_node != self.default_noise_node):
                # warn user that nodes differ
                LOGGER.warning("specified noise node (%s) is not the same as "
                               "circuit's default (%s)", noise_node,
                               self.default_noise_node)
        else:
            if self.default_noise_node is not None:
                # use default noise node
                self.noise_node = self.default_noise_node
                LOGGER.info("Using default noise node: %s", self.noise_node)

                compute_noise = True

        # check that we're solving something
        if not compute_tfs and not compute_noise:
            raise Exception("no solution requested (specify an input node, a "
                            "noise node, or both)")

        if compute_tfs:
            # signal results matrix
            tfs = self._results_matrix(n_freqs)

            # input vector
            y = self._results_matrix(1)

            # set input voltage
            y[self._input_index, 0] = 1

        if compute_noise:
            # noise results matrix
            noise = self._results_matrix(n_freqs)

        if print_progress:
            # update progress every 1% of the way there
            update = n_freqs // 100

            # if there are less than 100 frequencies, update progress after
            # every frequency
            if update == 0:
                update += 1

            # create frequency generator with progress bar
            freq_gen = _print_progress(frequencies, n_freqs, update=update,
                                       stream=progress_stream)
        else:
            # just use provided frequency sequence
            freq_gen = frequencies

        # frequency loop
        for freq_index, frequency in enumerate(freq_gen):
            # get matrix for this frequency
            matrix = self.matrix(frequency)

            if compute_tfs:
                # solve transfer functions
                tfs[:, freq_index] = spsolve(matrix, y)

            if compute_noise:
                noise[:, freq_index] = self._noise_at_node(self.noise_node,
                                                           matrix, frequency)

        # create solution
        return Solution(self, frequencies, tfs, noise, self.noise_node)

    def _noise_at_node(self, node, matrix, frequency):
        """Compute noise from components projected to the specified node

        :param node: node to project noise to
        :type node: :class:`~Node`
        :param matrix: circuit matrix
        :type matrix: :class:`scipy.sparse.spmatrix` or :class:`np.ndarray`
        :param frequency: frequency at which to compute noise
        :type frequency: float or Numpy scalar
        :return: vector containing noise from each circuit component/node
        :rtype: :class:`np.ndarray`
        """

        # create column vector
        e_n = self._results_matrix(1)
        e_n[self._voltage_node_matrix_index(node), 0] = 1

        # set input impedance
        # NOTE: this issues a SparseEfficiencyWarning
        index = self._component_matrix_index(Input())
        matrix[index, index] = self.input_impedance

        # solve, giving transfer function from each component/node to the
        # output (size = nx0)
        y_hat = spsolve(matrix.T, e_n)

        # noise current/voltage input vector (original size = nx1, squeezed
        # size = nx0)
        k = self._noise_input_vector(frequency)

        # noise at output (size = nx1)
        return np.abs(y_hat * k)

    def _noise_input_vector(self, frequency):
        """Create noise input vector at a given frequency

        Creates an nx1 column vector containing the voltage and current noise at
        each component and node, respectively.

        :param frequency: frequency at which to compute noise
        :type frequency: float or Numpy scalar
        :return: noise input column vector
        :rtype: :class:`~np.ndarray`
        """

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
    def _input_index(self):
        # FIXME: support current inputs
        return self._component_matrix_index(Input())

    @property
    def dim_size(self):
        """Circuit matrix dimension size

        :return: number of rows/columns in circuit matrix
        :rtype: int
        """

        return self.n_components + len(list(self.non_gnd_nodes))

    @property
    def _last_index(self):
        """Circuit matrix index corresponding to last row/column

        :return: last index
        :rtype: int
        """

        return self.dim_size - 1

    def _component_matrix_index(self, component):
        """Circuit matrix index corresponding to a component

        :param component: component to get index for
        :type component: :class:`~Component`
        :return: component index
        :rtype: int
        """

        return self.components.index(component)

    def _voltage_node_matrix_index(self, node):
        """Circuit matrix index corresponding to a voltage at a node

        This represents a matrix column.

        :param node: node to get index for
        :type node: :class:`~Node`
        :return: node index
        :rtype: int
        """

        return self.n_components + self.node_index(node)

    def _current_node_matrix_index(self, node):
        """Circuit matrix index corresponding to a current through a node

        This represents a matrix row.

        :param node: node to get index for
        :type node: :class:`~Node`
        :return: node index
        :rtype: int
        """

        return self.n_components + self.node_index(node)

    def _results_matrix(self, *depth):
        """Get empty matrix of specified size

        The results matrix always has n rows, where n is the number of
        components and nodes in the circuit. The column size, and the size of
        any additional dimensions, can be specified with subsequent ``depth``
        parameters.

        :param depth: size of index 1...x
        :type depth: int
        :return: empty results matrix
        :rtype: :class:`~np.ndarray`
        """

        return np.zeros((self.dim_size, *depth), dtype="complex64")

    def component_equations(self):
        """Get linear equations representing components in circuit

        :return: sequence of component equations
        :rtype: Generator[ComponentEquation]
        """

        return [component.equation() for component in self.components]

    def node_equations(self):
        """Get linear equations representing nodes in circuit

        :return: sequence of node equations
        :rtype: Generator[NodeEquation]
        """

        return [node.equation() for node in self.non_gnd_nodes]

    def node_index(self, node):
        """Get node serial number

        This does not include the ground node, so the first non-ground node
        has serial number 0.

        :param node: node
        :type node: :class:`~Node`
        :return: node serial number
        :rtype: int
        """

        if node == Gnd():
            raise ValueError("ground node does not have an index")

        return list(self.non_gnd_nodes).index(node)

    def coefficients(self):
        """Get circuit component and node equation coefficients

        :return: sequence of equation coefficients
        :rtype: Generator[BaseCoefficient]
        """

        for equation in self.equations:
            yield from equation.coefficients

    def print_equations(self, stream=sys.stdout, *args, **kwargs):
        """Pretty print circuit equations

        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        """

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
                    print("(%g%+gi)" % (np.real(element), np.imag(element)),
                          end="", file=stream)

                print(" %s " % self.formatted_element(column), end="", file=stream)

            if row == self._input_index:
                print(" = 1", file=stream)
            else:
                print(" = 0", file=stream)

    def print_matrix(self, stream=sys.stdout, *args, **kwargs):
        """Pretty print circuit matrix

        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        """

        # get matrix
        matrix = self.matrix(*args, **kwargs)

        # convert complex values to magnitudes (leaving others, like -1, alone)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.iscomplex(matrix[row, column]):
                    matrix[row, column] = np.abs(matrix[row, column])

        # remove imaginary parts (which are all zero) and convert to numpy array
        array = matrix.real.toarray()

        # tabulate data
        table = tabulate(array, self.column_headers,
                         tablefmt=CONF["format"]["table"])

        # output
        print(table, file=stream)

    @property
    def column_headers(self):
        """Get column headers for matrix elements"""

        return [self.formatted_element(n) for n in range(self.dim_size)]

    def formatted_element(self, index):
        """Format matrix element for pretty printer

        Determines if the specified ``index`` refers to a component, voltage
        or current node, and prints information accordingly.

        :param index: index to format
        :type index: int
        """

        if index < self.n_components:
            return "i[%s]" % self.components[index].name
        elif index <= self.dim_size:
            return "V[%s]" % list(self.non_gnd_nodes)[index - self.n_components]

        raise ValueError("invalid element index")
