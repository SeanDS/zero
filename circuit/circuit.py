"""Electronic circuit class to which linear components can be added and
on which simulations can be performed."""

import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import logging
import statistics
from tabulate import tabulate

HAS_GRAPHVIZ = False
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    pass

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
    """Create new complex-valued sparse matrix.

    Returns
    -------
    :class:`~lil_matrix`
        sparse matrix
    """

    # complex64 gives real and imaginary parts each represented as 32-bit floats
    # with 8 bits exponent and 23 bits mantissa, giving between 6 and 7 digits
    # of precision; good enough for most purposes
    return lil_matrix(dtype="complex128", *args, **kwargs)

def solve(A, b):
    """Solve linear system.

    Parameters
    ----------
    A : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
        square matrix
    B : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
        matrix or vector representing right hand side of matrix equation

    Returns
    -------
    solution : :class:`~np.ndarray`, :class:`~scipy.sparse.spmatrix`
        x in the equation Ax = b
    """

    # permute specification chosen to minimise error with LISO
    return spsolve(A, b, permc_spec="MMD_AT_PLUS_A")

class Circuit(object):
    """Represents an electronic circuit containing linear components.

    A circuit can contain components like :class:`resistors <.Resistor>`,
    :class:`capacitors <.Capacitor>`, :class:`inductors <.Inductor>` and
    :class:`op-amps <.OpAmp>`. These are added to the circuit via
    the :meth:`~Circuit.add_component` method.

    Attributes
    ----------
    components : sequence of :class:`components <.components.Component>`
        The circuit's components.
    nodes : sequence of :class:`nodes <.components.Node>`
        The circuit's nodes.
    prescale : :class:`bool`
        whether to prescale matrix elements into natural units for numerical
        precision purposes
    """

    def __init__(self):
        """Instantiate a new circuit"""

        # empty lists of components and nodes
        self.components = []
        self.nodes = []
        self.prescale = True

        # defaults
        self._noise_node = None

    @property
    def noise_node(self):
        """Circuit's noise node

        Returns
        -------
        :class:`~.components.Node`
            The circuit's noise node
        """

        return self._noise_node

    @noise_node.setter
    def noise_node(self, node):
        """Set circuit's noise node

        Parameters
        ----------
        node : :class:`~.components.Node`
            The circuit's new noise node
        """

        if not isinstance(node, Node):
            node = Node(node)

        self._noise_node = node

    @property
    def non_gnd_nodes(self):
        """Circuit's nodes, excluding ground

        Yields
        ------
        :class:`~.components.Node`
            Circuit nodes, excluding ground.
        """

        return [node for node in self.nodes if node is not Node("gnd")]

    @property
    def element_headers(self):
        """Headers corresponding to circuit's matrix elements.

        Yields
        ------
        :class:`str`
            column header
        """

        return [self.format_element(element) for element in self.elements]

    @property
    def element_names(self):
        """Names of elements within the circuit.

        Yields
        ------
        :class:`str`
            element name
        """

        return [element.name for element in self.elements]

    def format_element(self, element):
        """Formats matrix element.

        Determines if the specified `element` refers to a component current or a
        voltage node and prints information accordingly.

        Parameters
        ----------
        element : :class:`~.components.Component`, :class:`~.components.Node`
            element to format

        Raises
        ------
        ValueError
            if element type is not recognised
        """

        if isinstance(element, Component):
            return "i[%s]" % element.name
        elif isinstance(element, Node):
            return "V[%s]" % element.name

        raise ValueError("unrecognised element")

    @property
    def n_components(self):
        """Get number of components in circuit

        Returns
        -------
        :class:`int`
            number of components
        """

        return len(self.components)

    @property
    def n_nodes(self):
        """Get number of nodes in circuit

        Returns
        -------
        :class:`int`
            number of nodes
        """

        return len(self.nodes)

    def add_component(self, component):
        """Add component, and its nodes, to the circuit.

        Parameters
        ----------
        component : :class:`~.components.Component`
            component to add

        Raises
        ------
        ValueError
            if component is None, or already present in the circuit
        """

        if component is None:
            raise ValueError("component cannot be None")
        elif component in self.components:
            raise ValueError("component %s already in circuit" % component)

        # add component to end of list
        self.components.append(component)

        # register component's nodes
        for node in component.nodes:
            self._add_node(node)

    def add_input(self, *args, **kwargs):
        """Add input to circuit."""
        self.add_component(Input(*args, **kwargs))

    def add_resistor(self, *args, **kwargs):
        """Add resistor to circuit."""
        self.add_component(Resistor(*args, **kwargs))

    def add_capacitor(self, *args, **kwargs):
        """Add capacitor to circuit."""
        self.add_component(Capacitor(*args, **kwargs))

    def add_inductor(self, *args, **kwargs):
        """Add inductor to circuit."""
        self.add_component(Inductor(*args, **kwargs))

    def add_opamp(self, *args, **kwargs):
        """Add op-amp to circuit."""
        self.add_component(OpAmp(*args, **kwargs))

    def add_library_opamp(self, model, *args, **kwargs):
        """Add library op-amp to circuit."""
        data = LIBRARY.get_data(model)

        self.add_opamp(model=OpAmpLibrary.format_name(model), *args, **kwargs,
                       **data)

    def _add_node(self, node):
        """Add node to circuit.

        Parameters
        ----------
        node : :class:`~.components.Node`
            node to add

        Raises
        ------
        ValueError
            if the node is None
        """

        if node is None:
            raise ValueError("node cannot be None")

        if node not in self.nodes:
            # this is the first time this circuit has seen this node
            # reset its sources and sinks
            node.defaults()

            # add
            self.nodes.append(node)

    def get_component(self, component_name):
        """Get circuit component by name.

        Parameters
        ----------
        component_name : :class:`str`
            name of component to fetch

        Returns
        -------
        :class:`~.components.Component`
            component

        Raises
        ------
        ValueError
            if component not found
        """

        component_name = component_name.lower()

        for component in self.components:
            if component.name.lower() == component_name:
                return component

        raise ValueError("component %s not found" % component_name)

    def get_node(self, node_name):
        """Get circuit node by name.

        Parameters
        ----------
        node_name : :class:`str`
            name of node to fetch

        Returns
        -------
        :class:`~.components.Node`
            node

        Raises
        ------
        ValueError
            if node not found
        """

        node_name = node_name.lower()

        for node in self.nodes:
            if node.name.lower() == node_name:
                return node

        raise ValueError("node %s not found" % node_name)

    @property
    def resistors(self):
        """Circuit resistors.

        Yields
        ------
        :class:`~.components.Resistor`
            circuit resistors
        """

        return [component for component in self.components
                if isinstance(component, Resistor)]

    @property
    def capacitors(self):
        """Circuit capacitors.

        Yields
        ------
        :class:`~.components.Capacitor`
            circuit capacitors
        """

        return [component for component in self.components
                if isinstance(component, Capacitor)]

    @property
    def inductors(self):
        """Circuit inductors.

        Yields
        ------
        :class:`~.components.Inductor`
            circuit inductors
        """

        return [component for component in self.components
                if isinstance(component, Inductor)]

    @property
    def opamps(self):
        """Circuit op-amps.

        Yields
        ------
        :class:`~.components.OpAmp`
            circuit op-amps
        """

        return [component for component in self.components
                if isinstance(component, OpAmp)]

    def _get_tf_matrix(self, frequency):
        """Calculate and return matrix used to solve for circuit transfer \
        functions for a given frequency.

        This constructs a sparse matrix containing the voltage and current
        equations for each component and node.

        Parameters
        ----------
        frequency : Numpy scalar, :class:`float`
            frequency at which to calculate circuit impedances

        Returns
        -------
        :class:`scipy.sparse.spmatrix`
            circuit matrix

        Raises
        ------
        ValueError
            if an invalid coefficient type is encountered
        """

        # create new sparse matrix
        matrix = sparse((self.dim_size, self.dim_size))

        if self.prescale:
            scale = 1 / self.mean_resistance
        else:
            scale = 1

        # Kirchoff's voltage law / op-amp voltage gain equations
        for equation in self.component_equations:
            for coefficient in equation.coefficients:
                # default indices
                row = self._component_matrix_index(equation.component)
                column = self._component_matrix_index(equation.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                if isinstance(coefficient, ImpedanceCoefficient):
                    # don't change indices, but scale impedance
                    value *= scale
                elif isinstance(coefficient, VoltageCoefficient):
                    # includes extra I[in] column (at index == self.n_components)
                    column = self._node_matrix_index(coefficient.node)
                else:
                    raise ValueError("invalid coefficient type")

                matrix[row, column] = value

        # Kirchoff's current law
        for equation in self.node_equations:
            for coefficient in equation.coefficients:
                if not isinstance(coefficient, CurrentCoefficient):
                    raise ValueError("invalid coefficient type")

                row = self._node_matrix_index(equation.node)
                column = self._component_matrix_index(coefficient.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                matrix[row, column] = value

        return matrix

    def _get_noise_matrix(self, *args, **kwargs):
        """Calculate and return matrix used to solve for circuit noise at a \
        given frequency.

        Parameters
        ----------
        frequency : Numpy scalar, :class:`float`
            frequency at which to calculate circuit impedances

        Returns
        -------
        :class:`scipy.sparse.spmatrix`
            circuit matrix
        """

        # simply return the transpose of the transfer function matrix
        return self._get_tf_matrix(*args, **kwargs).T

    def calculate_tfs(self, frequencies, output_components=[], output_nodes=[],
                      stream=sys.stdout, print_equations=False,
                      print_matrix=False, *args, **kwargs):
        """Calculate circuit transfer functions from input \
        :class:`component <.Component>` / :class:`node <.Node>` to output \
        :class:`components <.Component>` / :class:`nodes <.Node>`.

        Parameters
        ----------
        frequencies : sequence of Numpy scalars or :class:`float`
            sequence of frequencies to solve circuit for
        output_components : sequence of :class:`~.Component` or :class:`str`, :class:`str`
            output components to calculate transfer functions to; specify "all"
            to compute all
        output_nodes : sequence of :class:`~.components.Node` or :class:`str`, :class:`str`
            output nodes to calculate transfer functions to; specify "all" to
            compute all
        stream : :class:`io.IOBase`
            stream to print to
        print_equations : :class:`bool`
            whether to print circuit equations
        print_matrix : :class:`bool`
            whether to print circuit matrix
        print_progress : :class:`bool`
            whether to print solve progress to stream

        Returns
        -------
        :class:`~.solution.Solution`
            solution

        Raises
        ------
        Exception
            if no input is present within the circuit
        ValueError
            if neither output components nor nodes are specified
        ValueError
            if input type is unrecognised
        """

        if not self.has_input:
            raise Exception("circuit must contain an input")

        # parse output parameters
        output_components = list(self._parse_component_list(output_components))
        output_nodes = list(self._parse_node_list(output_nodes))

        if not len(output_components) and not len(output_nodes):
            raise ValueError("specify output component(s), output node(s), or "
                             "both")

        # calculate transfer functions by solving the transfer matrix for input
        # at the circuit's input node/component
        tfs = self._invert(frequencies, self._get_tf_matrix, self._input_vector,
                           stream=stream, *args, **kwargs)

        # create solution
        solution = Solution(self, frequencies)

        # skipped tfs
        skips = []

        # output component indices
        for component in output_components:
            # extract transfer function for this component
            tf = tfs[self._component_matrix_index(component), :]

            if np.all(tf) == 0:
                # skip null transfer function
                skips.append(component)
                # skip this iteration
                continue

            # create data series
            series = Series(x=frequencies, y=tf)

            # create appropriate transfer function depending on input type
            if self.has_voltage_input:
                function = VoltageCurrentTF(source=self.input_component.node_p,
                                            sink=component, series=series)
            elif self.has_current_input:
                function = CurrentCurrentTF(source=self.input_component,
                                            sink=component, series=series)
            else:
                raise ValueError("specify either a current or voltage input")

            # add transfer function to solution
            solution.add_tf(function)

        # output node indices
        for node in output_nodes:
            # extract transfer function for this node
            tf = tfs[self._node_matrix_index(node), :]

            if np.all(tf) == 0:
                # skip null tf
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
                raise ValueError("specify either a current or voltage input")

            # add transfer function to solution
            solution.add_tf(function)

        if len(skips):
            LOGGER.info("skipped null elements: %s",
                        ", ".join([str(tf) for tf in skips]))

        if print_equations:
            self._print_equations(matrix=self._get_tf_matrix(frequency=1),
                                  rhs=self._input_vector, stream=stream)
        if print_matrix:
            self._print_matrix(matrix=self._get_tf_matrix(frequency=1),
                               rhs=self._input_vector, stream=stream)

        return solution

    def calculate_noise(self, frequencies, noise_node, stream=sys.stdout,
                        print_equations=False, print_matrix=False, *args,
                        **kwargs):
        """Calculate noise from circuit :class:`component <.Component>` / \
        :class:`node <.Node>` at a particular :class:`node <.Node>`.

        Parameters
        ----------
        frequencies : sequence of Numpy scalars or :class:`float`
            sequence of frequencies to solve circuit for
        noise_node : :class:`~.components.Node`, :class:`str`
            node to project noise to
        stream : :class:`io.IOBase`
            stream to print to
        print_equations : :class:`bool`
            whether to print circuit equations
        print_matrix : :class:`bool`
            whether to print circuit matrix
        print_progress : :class:`bool`
            whether to print solve progress to stream

        Returns
        -------
        :class:`~.solution.Solution`
            solution

        Raises
        ------
        Exception
            if no input is present within the circuit
        ValueError
            if unrecognised noise source is present in circuit
        """

        if not self.has_input:
            raise Exception("circuit must contain an input")

        if noise_node is not None:
            # use noise node specified in this call
            self.noise_node = noise_node

        # calculate noise functions by solving the transfer matrix for input
        # at the circuit's noise sources
        noise_matrix = self._invert(frequencies, self._get_noise_matrix,
                                    self._noise_vector, stream=stream, *args,
                                    **kwargs)

        # create solution
        solution = Solution(self, frequencies)

        # skipped noise sources
        skips = []

        # loop over circuit's noise sources
        for noise in self.noise_sources:
            # get this element's noise spectral density
            spectral_density = noise.spectral_density(frequencies=frequencies)

            if np.all(spectral_density) == 0:
                # skip null noise source
                skips.append(noise)
                # skip this iteration
                continue

            if isinstance(noise, ComponentNoise):
                # noise is from a component; use its matrix index
                index = self._component_matrix_index(noise.component)
            elif isinstance(noise, NodeNoise):
                # noise is from a node; use its matrix index
                index = self._node_matrix_index(noise.node)
            else:
                raise ValueError("unrecognised noise source present in circuit")

            # get response from this element to every other
            response = noise_matrix[index, :]

            # multiply response from element to noise node by noise entering
            # at that element, for all frequencies
            projected_noise = np.abs(response * spectral_density)

            # create series
            series = Series(x=frequencies, y=projected_noise)

            # add noise function to solution
            solution.add_noise(NoiseSpectrum(source=noise,
                                             sink=self.noise_node,
                                             series=series))

        if len(skips):
            LOGGER.info("skipped null noise sources: %s",
                        ", ".join([str(noise) for noise in skips]))

        if print_equations:
            self._print_equations(matrix=self._get_noise_matrix(frequency=1),
                                  rhs=self._noise_vector, stream=stream)
        if print_matrix:
            self._print_matrix(matrix=self._get_noise_matrix(frequency=1),
                               rhs=self._noise_vector, stream=stream)

        return solution

    @property
    def dim_size(self):
        """Circuit matrix dimension size

        Returns
        -------
        :class:`int`
            number of rows/columns in circuit matrix
        """

        # dimension size is the number of components added to circuit, including
        # the input, plus the number of non-ground nodes
        return self.n_components + len(list(self.non_gnd_nodes))

    def _results_matrix(self, *depth):
        """Get empty matrix of specified size

        The results matrix always has n rows, where n is the number of
        components and nodes in the circuit. The column size, and the size of
        any additional dimensions, can be specified with subsequent ``depth``
        parameters.

        Parameters
        ----------
        depth : :class:`int`
            size of index 1...x

        Returns
        -------
        :class:`~np.ndarray`
            empty results matrix
        """

        return np.zeros((self.dim_size, *depth), dtype="complex64")

    def _invert(self, frequencies, A_function, b, stream=sys.stdout,
                print_progress=True):
        """Solve matrix equation Ax = b.

        Parameters
        ----------
        frequencies : sequence of Numpy scalars or :class:`float`
            sequence of frequencies to solve circuit for
        A_function : callable
            callable which returns the matrix at a particular frequency; \
            must take as its only parameter a frequency
        b : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            right hand side of matrix equation
        stream : :class:`io.IOBase`
            stream to print to
        print_progress : bool
            whether to print solve progress to stream

        Returns
        -------
        :class:`~np.ndarray`
            inverse of A

        Raises
        ------
        ValueError
            if A_function is not callable
        """

        if not callable(A_function):
            raise ValueError("A_function must be a callable")

        # number of frequencies to calculate
        n_freqs = len(frequencies)

        # results matrix
        results = self._results_matrix(n_freqs)

        # scale vector, for converting units, if necessary
        scale = self._results_matrix(1).T
        scale[0, :] = 1

        if self.prescale:
            # convert currents from natural units back to amperes
            prescaler = 1 / self.mean_resistance

            for component in self.components:
                scale[0, self._component_matrix_index(component)] = prescaler

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
            results[:, index] = solve(matrix, b) * scale

        return results

    def _parse_component_list(self, components):
        """Parse the specified component list.

        The component list can be a sequence containing :class:`~.Component`
        objects or strings containing names of components in the circuit, or it
        can be equal to the string "all", in which case all circuit components
        will be returned.

        Parameters
        ----------
        components : sequence of :class:`~.components.Component` or :class:`str`, :class:`str`
            sequence of components or component names, or "all"

        Yields
        ------
        :class:`~.components.Component`
            parsed component
        """

        if components == "all":
            yield from self.components
            return

        # assume sequence was provided
        for component in components:
            if not isinstance(component, Component):
                # parse component name
                component = self.get_component(component)

            yield component

    def _parse_node_list(self, nodes):
        """Parse the specified node list

        The node list can be a sequence containing :class:`~Mode` objects or
        strings containing names of nodes in the circuit, or it can be equal to
        the string "all", in which case all circuit nodes will be returned.


        Parameters
        ----------
        nodes : sequence of :class:`~.components.Node` or :class:`str`, :class:`str`
            sequence of nodes or node names, or "all"

        Yields
        ------
        :class:`~.components.Node`
            parsed node
        """

        if nodes == "all":
            yield from self.non_gnd_nodes
            return

        # assume sequence was provided
        for node in nodes:
            if not isinstance(node, Node):
                # parse node name
                node = self.get_node(node)

            yield node

    @property
    def input_component(self):
        """Circuit's input component."""
        return self.get_component("input")

    @property
    def input_impedance(self):
        """Circuit's input impedance."""
        return self.input_component.impedance

    @property
    def noise_sources(self):
        """Noise sources in the circuit.

        Yields
        ------
        :class:`~.components.ComponentNoise`
            noise source
        """

        for component in self.components:
            yield from component.noise

    @property
    def has_noise_input(self):
        """Check if circuit has a noise input."""
        return self.input_component.input_type is Input.TYPE_NOISE

    @property
    def has_voltage_input(self):
        """Check if circuit has a voltage input."""
        return self.input_component.input_type is Input.TYPE_VOLTAGE

    @property
    def has_current_input(self):
        """Check if circuit has a current input."""
        return self.input_component.input_type is Input.TYPE_CURRENT

    @property
    def has_input(self):
        """Check if circuit has an input."""
        try:
            self.input_component
        except ValueError:
            return False

        return True

    @property
    def mean_resistance(self):
        """Average circuit resistance"""

        return statistics.mean([resistor.resistance for resistor in self.resistors])

    def _component_matrix_index(self, component):
        """Circuit matrix index corresponding to a component.

        Parameters
        ----------
        component : :class:`~.Component`
            component to get index for

        Returns
        -------
        :class:`int`
            component index
        """

        return self.components.index(component)

    def _node_matrix_index(self, node):
        """Circuit matrix index corresponding to a node.

        Parameters
        ----------
        node : :class:`~.components.Node`
            node to get index for

        Returns
        -------
        :class:`int`
            node index
        """

        return self.n_components + self.node_index(node)

    @property
    def _input_component_index(self):
        """Input component's matrix index."""
        return self._component_matrix_index(self.input_component)

    @property
    def _input_node_index(self):
        """Input node's matrix index."""
        return self._node_matrix_index(self.input_component.node2)

    @property
    def _noise_node_index(self):
        """Noise node's matrix index."""
        return self._node_matrix_index(self.noise_node)

    @property
    def _input_vector(self):
        """Circuit input vector.

        This creates a vector of size nx1, where n is the number of elements in
        the circuit, and sets the input component's coefficient to 1 before
        returning it.

        Returns
        -------
        :class:`~np.ndarray`
            circuit's input vector
        """

        # create column vector
        y = self._results_matrix(1)

        # set input to input component
        y[self._input_component_index, 0] = 1

        return y

    @property
    def _noise_vector(self):
        """Get circuit noise (output) vector

        This creates a vector of size nx1, where n is the number of elements in
        the circuit, and sets the noise node's coefficient to 1 before
        returning it.

        Returns
        -------
        :class:`~np.ndarray`
            circuit's noise output vector
        """

        # create column vector
        e_n = self._results_matrix(1)

        # set input to noise node
        e_n[self._noise_node_index, 0] = 1

        return e_n

    @property
    def component_equations(self):
        """Linear equations representing circuit components.

        Yields
        ------
        :class:`~.components.ComponentEquation`
            component equation
        """

        return [component.equation() for component in self.components]

    @property
    def node_equations(self):
        """Linear equations representing circuit nodes.

        Yields
        ------
        :class:`~.components.NodeEquation`
            sequence of node equations
        """

        return [node.equation() for node in self.non_gnd_nodes]

    def node_index(self, node):
        """Get node serial number.

        This does not include the ground node, so the first non-ground node
        has serial number 0.

        Parameters
        ----------
        node : :class:`~.Node`
            node

        Returns
        -------
        :class:`int`
            node serial number

        Raises
        ------
        ValueError
            if ground node is specified
        """

        if node == Node("gnd"):
            raise ValueError("ground node does not have an index")

        return list(self.non_gnd_nodes).index(node)

    def _print_equations(self, matrix, rhs, stream=sys.stdout):
        """Pretty print circuit equations.

        Parameters
        ----------
        matrix : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            left hand side of matrix equation, representing the circuit matrix
        rhs : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            right hand side of matrix equation, representing the input or output
            vector
        stream : :class:`io.IOBase`
            stream to print to
        """

        print("Circuit equations:", file=stream)
        for row, rhs_value in zip(range(matrix.shape[0]), rhs):
            # flag to suppress leading sign
            first = True

            for column, header in zip(range(matrix.shape[1]), self.element_headers):
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
        """Pretty print circuit matrix.

        Parameters
        ----------
        matrix : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            left hand side of matrix equation, representing the circuit matrix
        rhs : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            right hand side of matrix equation, representing the input or output
            vector
        stream : :class:`io.IOBase`
            stream to print to
        """

        # get matrix in full (non-sparse) format to allow stacking
        matrix = matrix.toarray()

        # attach right hand side to left hand side
        matrix = np.concatenate((matrix, rhs), axis=1)

        # convert complex values to magnitudes (leaving others, like -1, alone)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if np.iscomplex(matrix[row, column]):
                    matrix[row, column] = np.abs(matrix[row, column])

        # remove imaginary parts (which are all now zero)
        array = matrix.real

        # create vector of element names, with two dimensions
        element_names = np.expand_dims(np.array(self.element_names), axis=1)
        # prepend element names as first column
        array = np.concatenate((element_names, array), axis=1)

        # column headers, with extra columns for component names and RHS
        headers = [""] + self.element_headers + ["RHS"]

        # tabulate data
        table = tabulate(array, headers, tablefmt=CONF["format"]["table"])

        # output
        print("Circuit matrix:", file=stream)
        print(table, file=stream)

    @property
    def elements(self):
        """Matrix elements.

        Returns a sequence of elements - either components or nodes - in the
        order in which they appear in the matrix

        Yields
        ------
        :class:`~.components.Component`, :class:`~.components.Node`
            matrix elements
        """

        yield from self.components
        yield from self.non_gnd_nodes

    @property
    def element_names(self):
        """Names of elements (components and nodes) within the circuit.

        Yields
        ------
        :class:`str`
            matrix element names
        """

        return [element.name for element in self.elements]

    def format_element(self, element):
        """Format matrix element for pretty printer.

        Determines if the specified ``element`` refers to a component current
        or a voltage node and prints information accordingly.

        Parameters
        ----------
        element : :class:`~Component`, :class:`~Node`
            element to format

        Returns
        -------
        :class:`str`
            formatted element

        Raises
        ------
        ValueError
            if element is invalid
        """

        if isinstance(element, Component):
            return "i[%s]" % element.name
        elif isinstance(element, Node):
            return "V[%s]" % element.name

        raise ValueError("invalid element")

    def _node_graph(self):
        """Create Graphviz node graph."""

        if not HAS_GRAPHVIZ:
            raise NotImplementedError("Node graph representation requires the "
                                      "graphviz package")

        G = graphviz.Digraph(engine=CONF["graphviz"]["engine"])
        G.attr("node", style=CONF["graphviz"]["node_style"],
               fontname=CONF["graphviz"]["node_font_name"],
               fontsize=CONF["graphviz"]["node_font_size"])
        G.attr("edge", arrowhead=CONF["graphviz"]["edge_arrowhead"])
        G.attr("graph", splines=CONF["graphviz"]["graph_splines"],
               label="Made with graphviz and circuit.py",
               fontname=CONF["graphviz"]["graph_font_name"],
               fontsize=CONF["graphviz"]["graph_font_size"])
        node_map = {}

        def add_connection(C, conn, N):
            if N == 'gnd':
                G.node(C+'_'+N, shape='point', style='invis')
                G.edge(C+'_'+N, C+conn, dir='both', arrowtail='tee',
                       len='0.0', weight='10')
            else:
                if not N in node_map:
                    G.node(N, shape='point', xlabel=N,
                           width='0.1', fillcolor='Red')
                node_map[N] = N
                G.edge(node_map[N], C+conn)

        for C in self.components:
            connections = ['', '']
            if isinstance(C, OpAmp):
                # TODO: move these to components
                attr = {'shape': 'plain', 'margin': '0', 'orientation': '270'}
                attr['label'] = """<<TABLE BORDER="0" BGCOLOR="LightSkyBlue">
                    <TR><TD PORT="plus">+</TD><TD ROWSPAN="3">{0}<BR/>{1}</TD></TR>
                    <TR><TD> </TD></TR>
                    <TR><TD PORT="minus">-</TD></TR>
                </TABLE>>""".format(C.name, C.model)
                connections = [':plus', ':minus', ':e']
            elif isinstance(C, Inductor):
                attr = {'fillcolor': 'MediumSlateBlue', 'shape': 'diamond'}
            elif isinstance(C, Capacitor):
                attr = {'fillcolor': 'YellowGreen', 'shape': 'diamond'}
            elif isinstance(C, Resistor):
                attr = {'fillcolor': 'Orchid', 'shape': 'diamond'}
            elif isinstance(C, Input):
                attr = {'fillcolor': 'Orange',
                        'shape': ['ellipse','box','pentagon'][C.input_type-1]}
            else:
                print('Unrecognised element {0}: {1}'.format(C.name, C.__class__))

            G.node(C.name, **attr)
            for N, connection in zip(C.nodes, connections):
                add_connection(C.name, connection, N.name)

        return G

    if HAS_GRAPHVIZ:
        def _repr_svg_(self):
            """Graphviz rendering for Jupyter notebooks."""
            return self._node_graph()._repr_svg_()
