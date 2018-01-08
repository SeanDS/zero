"""Linear circuit simulations"""

import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import logging
from tabulate import tabulate

from .config import CircuitConfig, OpAmpLibrary
from .data import (VoltageVoltageTF, VoltageCurrentTF, CurrentCurrentTF,
                   CurrentVoltageTF, NoiseSpectrum, Series)
from .misc import _print_progress
from .components import (Component, Resistor, Capacitor, Inductor, OpAmp,
                         Input, Node, ComponentNoise, NodeNoise,
                         ImpedanceCoefficient, CurrentCoefficient,
                         VoltageCoefficient)
from .solution import Solution

LOGGER = logging.getLogger("circuit")
CONF = CircuitConfig()
LIBRARY = OpAmpLibrary()

def sparse(*args, **kwargs):
    """Create new complex-valued sparse matrix

    :return: sparse matrix
    :rtype: :class:`~lil_matrix`
    """

    # complex64 gives real and imaginary parts each represented as 32-bit floats
    # with 8 bits exponent and 23 bits mantissa, giving between 6 and 7 digits
    # of precision; good enough for most purposes
    return lil_matrix(dtype="complex64", *args, **kwargs)

def solve(A, b):
    """Solve linear system

    :param A: square matrix
    :type A: :class:`~np.ndarray` or :class:`~scipy.sparse.spmatrix`
    :param b: matrix or vector representing right hand side of matrix equation
    :type b: :class:`~np.ndarray` or :class:`~scipy.sparse.spmatrix`
    :return: solution
    :rtype: :class:`~np.ndarray` or :class:`~scipy.sparse.spmatrix`
    """

    # permute specification chosen to minimise error with LISO
    return spsolve(A, b, permc_spec="MMD_AT_PLUS_A")

class Circuit(object):
    """Represents an electronic circuit containing linear components"""

    def __init__(self):
        """Instantiate a new circuit

        A circuit can contain linear components like resistors, capacitors,
        inductors and op-amps.
        """

        # solver parameters
        self._noise_node = None
        self.components = []
        self.nodes = []
        self._matrix = None

    @property
    def noise_node(self):
        return self._noise_node

    @noise_node.setter
    def noise_node(self, node):
        if not isinstance(node, Node):
            node = Node(node)

        self._noise_node = node

    @property
    def non_gnd_nodes(self):
        """Get nodes in circuit, excluding ground

        :return: non-ground nodes
        :rtype: Generator[:class:`~Node`]
        """

        gnd = Node("gnd")

        for node in self.nodes:
            if node is not gnd:
                yield node

    @property
    def n_components(self):
        """Get number of components in circuit

        :return: number of components
        :rtype: int
        """

        return len(self.components)

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
            raise ValueError("component %s already in circuit" % component)

        # add component to end of list
        self.components.append(component)

        # register component's nodes
        for node in component.nodes:
            self.add_node(node)

        # reset matrix
        self._matrix = None

    def add_input(self, *args, **kwargs):
        self.add_component(Input(*args, **kwargs))

    def add_resistor(self, *args, **kwargs):
        self.add_component(Resistor(*args, **kwargs))

    def add_capacitor(self, *args, **kwargs):
        self.add_component(Capacitor(*args, **kwargs))

    def add_inductor(self, *args, **kwargs):
        self.add_component(Inductor(*args, **kwargs))

    def add_opamp(self, *args, **kwargs):
        self.add_component(OpAmp(*args, **kwargs))

    def add_library_opamp(self, model, *args, **kwargs):
        data = LIBRARY.get_data(model)

        self.add_opamp(model=OpAmpLibrary.format_name(model), *args, **kwargs,
                       **data)

    def add_node(self, node):
        """Add node to circuit

        :param node: node to add
        :type node: :class:`~Node`
        :raises Exception: if one of the component's nodes is unspecified
        """

        if node is None:
            raise Exception("Node cannot be none")

        if node not in self.nodes:
            # this is the first time this circuit has seen this node
            # reset its sources and sinks
            node.defaults()

            # add
            self.nodes.append(node)

    def get_component(self, component_name):
        """Get circuit component by name

        :param component_name: name of component to fetch
        :type component_name: str
        :return: component
        :rtype: :class:`~Component`
        :raises ValueError: if component not found
        """

        component_name = component_name.lower()

        for component in self.components:
            if component.name.lower() == component_name:
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

        node_name = node_name.lower()

        for node in self.nodes:
            if node.name.lower() == node_name:
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
        for equation in self.component_equations:
            for coefficient in equation.coefficients:
                # default indices
                row = self._component_matrix_index(equation.component)
                column = self._component_matrix_index(equation.component)

                if isinstance(coefficient, ImpedanceCoefficient):
                    # don't change indices
                    pass
                elif isinstance(coefficient, VoltageCoefficient):
                    # includes extra I[in] column (at index == self.n_components)
                    column = self._node_matrix_index(coefficient.node)
                else:
                    raise ValueError("invalid coefficient type")

                # fetch (potentially frequency-dependent) impedance
                if callable(coefficient.value):
                    # copy function
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    # copy value
                    self._matrix[row, column] = coefficient.value

        # Kirchoff's current law
        for equation in self.node_equations:
            for coefficient in equation.coefficients:
                if not isinstance(coefficient, CurrentCoefficient):
                    raise ValueError("invalid coefficient type")

                row = self._node_matrix_index(equation.node)
                column = self._component_matrix_index(coefficient.component)

                if callable(coefficient.value):
                    self._matrix_callables[(row, column)] = coefficient.value
                else:
                    self._matrix[row, column] = coefficient.value

    def _tf_matrix(self, frequency):
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

        return matrix

    def _noise_matrix(self, *args, **kwargs):
        return self._tf_matrix(*args, **kwargs).T

    def calculate_tfs(self, frequencies, output_components=[], output_nodes=[],
                  stream=sys.stdout, print_equations=False, print_matrix=False,
                  *args, **kwargs):
        """Calculate circuit transfer functions from input component/node to \
           output components/nodes

        :param frequencies: sequence of frequencies to solve circuit for
        :type frequencies: Sequence[Numpy scalar or float]
        :param output_components: output components to calculate transfer \
                                  functions to; specify "all" to compute all
        :type output_components: List[:class:`~Component` or str] or str
        :param output_nodes: output nodes to calculate transfer functions to; \
                             specify "all" to compute all
        :type output_nodes: List[:class:`~Node` or str] or str
        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        :param print_equations: whether to print circuit equations
        :type print_equations: bool
        :param print_matrix: whether to print circuit matrix
        :type print_matrix: bool
        :param print_progress: whether to print solve progress to stream
        :type print_progress: bool
        :return: solution
        :rtype: :class:`~Solution`
        """

        if not self.has_input:
            raise Exception("circuit must contain an input")

        # handle outputs
        if output_components == "all":
            output_components = list(self.components)
        else:
            output_components = list(output_components)
            for index, component in enumerate(output_components):
                if not isinstance(component, Component):
                    # parse component name
                    output_components[index] = self.get_component(component)
        if output_nodes == "all":
            output_nodes = list(self.non_gnd_nodes)
        else:
            output_nodes = list(output_nodes)
            for index, node in enumerate(output_nodes):
                if not isinstance(node, Node):
                    # parse node name
                    output_nodes[index] = self.get_node(node)

        if not len(output_components) and not len(output_nodes):
            raise ValueError("no outputs specified")

        # calculate transfer functions
        tfs = self._invert(frequencies, self._tf_matrix, self._input_vector,
                           stream=stream, *args, **kwargs)

        # create solution
        solution = Solution(self, frequencies)

        # skipped tfs
        skips = []

        # output component indices
        for component in output_components:
            # transfer function
            tf = tfs[self._component_matrix_index(component), :]

            if np.all(tf) == 0:
                # skip zero tf
                skips.append(component)

                # skip this iteration
                continue

            # create series
            series = Series(x=frequencies, y=tf)
            # create appropriate transfer function depending on input type
            if self.has_voltage_input:
                function = VoltageCurrentTF(source=self.input_component.node_p,
                                            sink=component, series=series)
            elif self.has_current_input:
                function = CurrentCurrentTF(source=self.input_component,
                                            sink=component, series=series)
            else:
                raise ValueError("unsupported input type")
            # add transfer function
            solution.add_tf(function)

        # output node indices
        for node in output_nodes:
            # transfer function
            tf = tfs[self._node_matrix_index(node), :]

            if np.all(tf) == 0:
                # skip zero tf
                skips.append(node)

                # skip this iteration
                continue

            # create series
            series = Series(x=frequencies, y=tf)

            # create appropriate transfer function depending on input type
            if self.has_voltage_input:
                function = VoltageVoltageTF(source=self.input_component.node_p,
                                            sink=node, series=series)
            elif self.has_current_input:
                function = CurrentVoltageTF(source=self.input_component,
                                            sink=node, series=series)
            else:
                raise ValueError("unrecognised input type")
            # add transfer function
            solution.add_tf(function)

        if len(skips):
            LOGGER.info("skipped null output nodes: %s",
                        ", ".join([str(tf) for tf in skips]))

        if print_equations:
            self._print_equations(matrix=self._tf_matrix(frequency=1),
                                  rhs=self._input_vector, stream=stream)
        if print_matrix:
            self._print_matrix(matrix=self._tf_matrix(frequency=1),
                               rhs=self._input_vector, stream=stream)

        return solution

    def calculate_noise(self, frequencies, noise_node, stream=sys.stdout,
                       print_equations=False, print_matrix=False, *args,
                       **kwargs):
        """Calculate noise from circuit components/nodes at a particular node

        :param frequencies: sequence of frequencies to solve circuit for
        :type frequencies: Sequence[Numpy scalar or float]
        :param noise_node: node to project noise to
        :type noise_node: :class:`~Node` or str
        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        :param print_equations: whether to print circuit equations
        :type print_equations: bool
        :param print_matrix: whether to print circuit matrix
        :type print_matrix: bool
        :param print_progress: whether to print solve progress to stream
        :type print_progress: bool
        :return: solution
        :rtype: :class:`~Solution`
        """

        if not self.has_input:
            raise Exception("circuit must contain an input")

        if noise_node is not None:
            # use noise node specified in this call
            self.noise_node = noise_node

        # calculate noise
        noise_matrix = self._invert(frequencies, self._noise_matrix,
                                    self._noise_vector, stream=stream, *args,
                                    **kwargs)

        # create solution
        solution = Solution(self, frequencies)

        # skipped noise sources
        skips = []

        for noise in self.noise_sources:
            # noise spectral density
            spectral_density = noise.spectral_density(frequencies=frequencies)

            if np.all(spectral_density) == 0:
                # skip zero noise source
                skips.append(noise)

                # skip this iteration
                continue

            if isinstance(noise, ComponentNoise):
                # matrix component index
                index = self._component_matrix_index(noise.component)
            elif isinstance(noise, NodeNoise):
                # matrix node index
                index = self._node_matrix_index(noise.node)
            else:
                raise ValueError("unrecognised noise")

            # response for this element
            response = noise_matrix[index, :]

            # multiply response from element to noise node by noise entering
            # at that element, for all frequencies
            projected_noise = np.abs(response * spectral_density)

            # create series
            series = Series(x=frequencies, y=projected_noise)

            solution.add_noise(NoiseSpectrum(source=noise,
                                             sink=self.noise_node,
                                             series=series))

        if len(skips):
            LOGGER.info("skipped null noise sources: %s",
                        ", ".join([str(noise) for noise in skips]))

        if print_equations:
            self._print_equations(matrix=self._noise_matrix(frequency=1),
                                  rhs=self._noise_vector, stream=stream)
        if print_matrix:
            self._print_matrix(matrix=self._noise_matrix(frequency=1),
                               rhs=self._noise_vector, stream=stream)

        return solution

    def _invert(self, frequencies, A_function, b, print_progress=True,
                stream=sys.stdout):
        if not callable(A_function):
            raise ValueError("A_function must be a callable")

        # number of frequencies to calculate
        n_freqs = len(frequencies)

        # results matrix
        results = self._results_matrix(n_freqs)

        if print_progress:
            # update progress every 1% of the way there
            update = n_freqs // 100

            # if there are less than 100 frequencies, update progress after
            # every frequency
            if update == 0:
                update += 1

            # create frequency generator with progress bar
            freq_gen = _print_progress(frequencies, n_freqs, update=update,
                                       stream=stream)
        else:
            # just use provided frequency sequence
            freq_gen = frequencies

        # frequency loop
        for index, frequency in enumerate(freq_gen):
            # get matrix for this frequency, converting to CSR format for
            # efficient solving
            matrix = A_function(frequency).tocsr()

            # call solver function
            results[:, index] = solve(matrix, b)

        return results

    @property
    def input_component(self):
        return self.get_component("input")

    @property
    def input_impedance(self):
        return self.input_component.impedance

    @property
    def has_noise_input(self):
        return self.input_component.input_type is Input.TYPE_NOISE

    @property
    def has_voltage_input(self):
        return self.input_component.input_type is Input.TYPE_VOLTAGE

    @property
    def has_current_input(self):
        return self.input_component.input_type is Input.TYPE_CURRENT

    @property
    def has_input(self):
        try:
            self.input_component
        except ValueError:
            return False

        return True

    @property
    def _input_component_index(self):
        return self._component_matrix_index(self.input_component)

    @property
    def _input_node_index(self):
        return self._node_matrix_index(self.input_component.node2)

    @property
    def _noise_node_index(self):
        return self._node_matrix_index(self.noise_node)

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

    @property
    def noise_sources(self):
        """Noise sources in the circuit"""

        for component in self.components:
            yield from component.noise

    def _component_matrix_index(self, component):
        """Circuit matrix index corresponding to a component

        :param component: component to get index for
        :type component: :class:`~Component`
        :return: component index
        :rtype: int
        """

        return self.components.index(component)

    def _node_matrix_index(self, node):
        """Circuit matrix index corresponding to a node

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

    @property
    def _input_vector(self):
        # create column vector
        y = self._results_matrix(1)

        # set input to input component
        y[self._input_component_index, 0] = 1

        return y

    @property
    def _noise_vector(self):
        # create column vector
        e_n = self._results_matrix(1)

        # set input to noise node
        e_n[self._noise_node_index, 0] = 1

        return e_n

    @property
    def component_equations(self):
        """Get linear equations representing components in circuit

        :return: sequence of component equations
        :rtype: Generator[ComponentEquation]
        """

        return [component.equation() for component in self.components]

    @property
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

        if node == Node("gnd"):
            raise ValueError("ground node does not have an index")

        return list(self.non_gnd_nodes).index(node)

    def coefficients(self):
        """Get circuit component and node equation coefficients

        :return: sequence of equation coefficients
        :rtype: Generator[BaseCoefficient]
        """

        for equation in self.equations:
            yield from equation.coefficients

    def _print_equations(self, matrix, rhs, stream=sys.stdout):
        """Pretty print circuit equations

        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        """

        print("Circuit equations:", file=stream)
        for row, rhs_value in zip(range(matrix.shape[0]), rhs):
            # flag to suppress leading sign
            first = True

            for column, header in zip(range(matrix.shape[1]), self.column_headers):
                element = matrix[row, column]

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

                print(header, end="", file=stream)
            print(" = %s" % rhs_value, file=stream)

    def _print_matrix(self, matrix, rhs, stream=sys.stdout, *args, **kwargs):
        """Pretty print circuit matrix

        :param stream: stream to print to
        :type stream: :class:`io.IOBase`
        """

        # get matrix in full (non-sparse) format to allow stacking
        matrix = matrix.toarray()

        # attach input vector on right hand side
        matrix = np.concatenate((matrix, rhs), axis=1)

        # convert complex values to magnitudes (leaving others, like -1, alone)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.iscomplex(matrix[row, column]):
                    matrix[row, column] = np.abs(matrix[row, column])

        # remove imaginary parts (which are all zero)
        array = matrix.real

        # prepend element names as first column
        element_names = np.expand_dims(np.array(self.element_names), axis=1)
        array = np.concatenate((element_names, array), axis=1)

        # tabulate data
        table = tabulate(array, [""] + self.column_headers + ["RHS"],
                         tablefmt=CONF["format"]["table"])

        # output
        print("Circuit matrix:", file=stream)
        print(table, file=stream)

    @property
    def column_headers(self):
        """Get column headers for matrix elements"""

        return [self.format_element(element)
                for element in self.elements]

    @property
    def elements(self):
        """Matrix elements

        Returns a sequence of elements - either components or nodes - in the
        order in which they appear in the matrix

        :return: elements
        :rtype: Generator[:class:`~Component` or :class:`~Node`]
        """

        yield from self.components
        yield from self.non_gnd_nodes

    @property
    def element_names(self):
        return [element.name for element in self.elements]

    def format_element(self, element):
        """Format matrix element for pretty printer

        Determines if the specified ``element`` refers to a component current
        or a voltage node and prints information accordingly.

        :param element: element to format
        :type element: :class:`~Component` or :class:`~Node`
        """

        if isinstance(element, Component):
            return "i[%s]" % element.name
        elif isinstance(element, Node):
            return "V[%s]" % element.name

        raise ValueError("invalid element")
