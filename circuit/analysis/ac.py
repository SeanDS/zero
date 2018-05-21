import abc
import sys
import logging
import numpy as np
from collections import defaultdict

from .base import BaseAnalysis
from ..config import CircuitConfig
from ..solve import Solver
from ..components import Component, Input, Node, ComponentNoise, NodeNoise
from ..data import (VoltageVoltageTF, VoltageCurrentTF, CurrentCurrentTF,
                    CurrentVoltageTF, NoiseSpectrum, Series)
from ..solution import Solution
from ..display import MatrixDisplay, EquationDisplay

# FIXME: move this into base analysis, make `progress_generator` function
from ..misc import _print_progress

LOGGER = logging.getLogger("ac-analysis")
CONF = CircuitConfig()

class SmallSignalAcAnalysis(BaseAnalysis):
    """Small signal circuit analysis"""

    def __init__(self, prescale=True, *args, **kwargs):
        super(SmallSignalAcAnalysis, self).__init__(*args, **kwargs)

        self.prescale = prescale

        # create solver
        self.solver = Solver()

        # empty fields
        self._noise_node = None
        self._node_sources = None
        self._node_sinks = None

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
    def dim_size(self):
        """Circuit matrix dimension size

        Returns
        -------
        :class:`int`
            number of rows/columns in circuit matrix
        """

        # dimension size is the number of components added to circuit, including
        # the input, plus the number of non-ground nodes
        return self.circuit.n_components + len(list(self.circuit.non_gnd_nodes))

    def get_empty_results_matrix(self, *depth):
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
        mixed
            empty results matrix
        """
        return self.solver.full((self.dim_size, *depth))

    def calculate_tf_matrix(self, frequency):
        """Calculate and return matrix used to solve for circuit transfer \
        functions for a given frequency

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
        matrix = self.solver.sparse((self.dim_size, self.dim_size))

        if self.prescale:
            scale = 1 / self.mean_resistance
        else:
            scale = 1
        
        # add sources and sinks
        self.set_up_sources_and_sinks()

        # Kirchoff's voltage law / op-amp voltage gain equations
        for equation in self.component_equations:
            for coefficient in equation.coefficients:
                # default indices
                row = self.component_matrix_index(equation.component)
                column = self.component_matrix_index(equation.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                if isinstance(coefficient, ImpedanceCoefficient):
                    # don't change indices, but scale impedance
                    value *= scale
                elif isinstance(coefficient, VoltageCoefficient):
                    # includes extra I[in] column (at index == self.n_components)
                    column = self.node_matrix_index(coefficient.node)
                else:
                    raise ValueError("invalid coefficient type")

                matrix[row, column] = value

        # Kirchoff's current law
        for equation in self.node_equations:
            for coefficient in equation.coefficients:
                if not isinstance(coefficient, CurrentCoefficient):
                    raise ValueError("invalid coefficient type")

                row = self.node_matrix_index(equation.node)
                column = self.component_matrix_index(coefficient.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                matrix[row, column] = value

        return matrix

    def calculate_noise_matrix(self, *args, **kwargs):
        """Calculate and return matrix used to solve for circuit noise at a \
        given frequency

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
        return self.calculate_tf_matrix(*args, **kwargs).T

    @property
    def input_vector(self):
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
        y = self.get_empty_results_matrix(1)

        # set input to input component
        y[self.input_component_index, 0] = 1

        return y

    @property
    def noise_vector(self):
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
        e_n = self.get_empty_results_matrix(1)

        # set input to noise node
        e_n[self.noise_node_index, 0] = 1

        return e_n

    def calculate_tfs(self, frequencies, output_components=[], output_nodes=[],
                      stream=sys.stdout, print_equations=False, print_matrix=False,
                      *args, **kwargs):
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

        if not self.circuit.has_input:
            raise Exception("circuit must contain an input")

        # parse output parameters
        output_components = list(self._parse_component_list(output_components))
        output_nodes = list(self._parse_node_list(output_nodes))

        if not len(output_components) and not len(output_nodes):
            raise ValueError("specify output component(s), output node(s), or "
                             "both")

        # calculate transfer functions by solving the transfer matrix for input
        # at the circuit's input node/component
        tfs = self.solve(frequencies, self.calculate_tf_matrix, self.input_vector,
                         stream=stream, *args, **kwargs)

        # create solution
        solution = Solution(self.circuit, frequencies)

        # skipped tfs
        skips = []

        # output component indices
        for component in output_components:
            # extract transfer function for this component
            tf = tfs[self.component_matrix_index(component), :]

            if np.all(tf) == 0:
                # skip null transfer function
                skips.append(component)
                # skip this iteration
                continue

            # create data series
            series = Series(x=frequencies, y=tf)

            # create appropriate transfer function depending on input type
            if self.has_voltage_input:
                function = VoltageCurrentTF(source=self.circuit.input_component.node_p,
                                            sink=component, series=series)
            elif self.has_current_input:
                function = CurrentCurrentTF(source=self.circuit.input_component,
                                            sink=component, series=series)
            else:
                raise ValueError("specify either a current or voltage input")

            # add transfer function to solution
            solution.add_tf(function)

        # output node indices
        for node in output_nodes:
            # extract transfer function for this node
            tf = tfs[self.node_matrix_index(node), :]

            if np.all(tf) == 0:
                # skip null tf
                skips.append(node)
                # skip this iteration
                continue

            # create series
            series = Series(x=frequencies, y=tf)

            # create appropriate transfer function depending on input type
            if self.has_voltage_input:
                function = VoltageVoltageTF(source=self.circuit.input_component.node_p,
                                            sink=node, series=series)
            elif self.has_current_input:
                function = CurrentVoltageTF(source=self.circuit.input_component,
                                            sink=node, series=series)
            else:
                raise ValueError("specify either a current or voltage input")

            # add transfer function to solution
            solution.add_tf(function)

        if len(skips):
            LOGGER.info("skipped null elements: %s",
                        ", ".join([str(tf) for tf in skips]))

        if print_equations:
            print(self.circuit_equations(matrix=self.calculate_tf_matrix(frequency=1),
                                         rhs=self.input_vector), file=stream)
        if print_matrix:
            print(self.circuit_matrix(matrix=self.calculate_tf_matrix(frequency=1),
                                      rhs=self.input_vector), file=stream)

        return solution

    def calculate_noise(self, frequencies, noise_node, stream=sys.stdout,
                        print_equations=False, print_matrix=False, *args, **kwargs):
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

        if not self.circuit.has_input:
            raise Exception("circuit must contain an input")

        if noise_node is not None:
            # use noise node specified in this call
            self.noise_node = noise_node

        # calculate noise functions by solving the transfer matrix for input
        # at the circuit's noise sources
        noise_matrix = self.solve(frequencies, self.calculate_noise_matrix,
                                  self.noise_vector, stream=stream, *args,
                                  **kwargs)

        # create solution
        solution = Solution(self.circuit, frequencies)

        # skipped noise sources
        skips = []

        # loop over circuit's noise sources
        for noise in self.circuit.noise_sources:
            # get this element's noise spectral density
            spectral_density = noise.spectral_density(frequencies=frequencies)

            if np.all(spectral_density) == 0:
                # skip null noise source
                skips.append(noise)
                # skip this iteration
                continue

            if isinstance(noise, ComponentNoise):
                # noise is from a component; use its matrix index
                index = self.component_matrix_index(noise.component)
            elif isinstance(noise, NodeNoise):
                # noise is from a node; use its matrix index
                index = self.node_matrix_index(noise.node)
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
            print(self.circuit_equations(matrix=self.calculate_noise_matrix(frequency=1),
                                         rhs=self.noise_vector), file=stream)
        if print_matrix:
            print(self.circuit_matrix(matrix=self.calculate_noise_matrix(frequency=1),
                                      rhs=self.noise_vector), file=stream)

        return solution

    def solve(self, frequencies, A_function, b, stream=sys.stdout,
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
        results = self.get_empty_results_matrix(n_freqs)

        # scale vector, for converting units, if necessary
        scale = self.get_empty_results_matrix(1).T
        scale[0, :] = 1

        if self.prescale:
            # convert currents from natural units back to amperes
            prescaler = 1 / self.mean_resistance

            for component in self.circuit.components:
                scale[0, self.component_matrix_index(component)] = prescaler

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
            results[:, index] = self.solver.solve(matrix, b) * scale

        return results

    def reset_sources_and_sinks(self):
        """Reset circuit's sources and sinks"""

        # dicts containing sets by default
        self._node_sources = defaultdict(set)
        self._node_sinks = defaultdict(set)

    def set_up_sources_and_sinks(self):
        """Set up circuit's sources and sinks
        
        This inspects the circuit's components and informs the circuit's nodes
        about current inputs and outputs. Nodes cannot generate their own
        equations unless they know about
        """

        # reset first
        self.reset_sources_and_sinks()

        # list of single-input, single-output components
        siso_components = list(self.circuit.passive_components) + [self.circuit.input_component]

        # single-input, single-output components sink to their first node and source from
        # their second
        for component in siso_components:
            if component.node1:
                self._node_sinks[component.node1].add(component) # current flows into here...
            if component.node2:
                self._node_sources[component.node2].add(component) # ...and out of here
        
        # op-amps source current from their third (output) node (their input nodes are
        # ideal and therefore don't source or sink current)
        for component in self.circuit.opamps:
            self._node_sources[component.node3].add(component) # current flows out of here

    def component_equation(self, component):
        """Equation representing circuit component

        The component equation represents the component in the circuit matrix. It maps
        input or noise sources to voltages across :class:`nodes <Node>` and currents
        through :class:`components <Component>` in terms of their
        :class:`impedance <ImpedanceCoefficient>` and
        :class:`voltage <VoltageCoefficient>` coefficients.

        Returns
        -------
        :class:`~ComponentEquation`
            Component equation containing :class:`coefficients <BaseCoefficient>`.
        """

        # nodal potential equation coefficients
        # impedance * current + voltage = 0
        coefficients = []

        if hasattr(component, "input_type"):
            # this is an input component
            if component.input_type in ["noise", "current"]:
                # set source impedance
                coefficients.append(ImpedanceCoefficient(component=component,
                                                         value=component.impedance))
            
            if component.input_type in ["noise", "voltage"]:
                # set sink node coefficient
                if component.node_n is not Node("gnd"):
                    # voltage
                    coefficients.append(VoltageCoefficient(node=component.node_n,
                                                           value=-1))

                # set source node coefficient
                if component.node_p is not Node("gnd"):
                    # voltage
                    coefficients.append(VoltageCoefficient(node=component.node_p,
                                                           value=1))
        else:
            # add input node coefficient
            if component.node1 is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=component.node1,
                                                       value=-1))

            # add output node coefficient
            if component.node2 is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=component.node2,
                                                       value=1))

            if hasattr(component, "gain"):
                # this is an op-amp
                # add voltage gain V[n3] = H(s) (V[n1] - V[n2])
                coefficients.append(VoltageCoefficient(node=component.node3,
                                                       value=component.inverse_gain))
            else:
                # this is a passive component
                # add impedance
                coefficients.append(ImpedanceCoefficient(component=component,
                                                         value=component.impedance))

        # create and return equation
        return ComponentEquation(component, coefficients=coefficients)

    def node_equation(self, node):
        """Equation representing circuit node
        
        This should be called after :method:`set_up_sources_and_sinks`.
        """

        # nodal current equation coefficients
        # current out - current in = 0
        coefficients = []

        for source in self._node_sources[node]:
            # add source coefficient
            if source is not Node("gnd"):
                coefficients.append(CurrentCoefficient(component=source,
                                                       value=1))

        for sink in self._node_sinks[node]:
            # add sink coefficient
            if sink is not Node("gnd"):
                coefficients.append(CurrentCoefficient(component=sink,
                                                       value=-1))

        # create and return equation
        return NodeEquation(node, coefficients=coefficients)

    @property
    def component_equations(self):
        """Linear equations representing circuit components

        Yields
        ------
        :class:`~.components.ComponentEquation`
            component equation
        """

        return [self.component_equation(component) for component in self.circuit.components]

    @property
    def node_equations(self):
        """Linear equations representing circuit nodes

        Yields
        ------
        :class:`~.components.NodeEquation`
            sequence of node equations
        """

        return [self.node_equation(node) for node in self.circuit.non_gnd_nodes]

    def component_matrix_index(self, component):
        """Circuit matrix index corresponding to a component

        Parameters
        ----------
        component : :class:`~.Component`
            component to get index for

        Returns
        -------
        :class:`int`
            component index
        """

        return self.component_index(component)

    def node_matrix_index(self, node):
        """Circuit matrix index corresponding to a node

        Parameters
        ----------
        node : :class:`~.components.Node`
            node to get index for

        Returns
        -------
        :class:`int`
            node index
        """

        return self.circuit.n_components + self.node_index(node)

    @property
    def input_component_index(self):
        """Input component's matrix index"""
        return self.component_matrix_index(self.circuit.input_component)

    @property
    def input_node_index(self):
        """Input node's matrix index"""
        return self.node_matrix_index(self.circuit.input_component.node2)

    @property
    def noise_node_index(self):
        """Noise node's matrix index"""
        return self.node_matrix_index(self.noise_node)

    @property
    def has_noise_input(self):
        """Check if circuit has a noise input."""
        return self.circuit.input_component.input_type == "noise"

    @property
    def has_voltage_input(self):
        """Check if circuit has a voltage input."""
        return self.circuit.input_component.input_type == "voltage"

    @property
    def has_current_input(self):
        """Check if circuit has a current input."""
        return self.circuit.input_component.input_type == "current"

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
            yield from self.circuit.components
            return

        # assume sequence was provided
        for component in components:
            if not isinstance(component, Component):
                # parse component name
                component = self.circuit.get_component(component)

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
            yield from self.circuit.non_gnd_nodes
            return

        # assume sequence was provided
        for node in nodes:
            if not isinstance(node, Node):
                # parse node name
                node = self.circuit.get_node(node)

            yield node

    def format_element(self, element):
        """Format matrix element for pretty printing.

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

    @property
    def element_headers(self):
        """Headers corresponding to circuit's matrix elements.

        Yields
        ------
        :class:`str`
            column header
        """

        return [self.format_element(element) for element in self.elements]

    def circuit_equations(self, matrix, rhs):
        """Get circuit equations

        Parameters
        ----------
        matrix : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            left hand side of matrix equation, representing the circuit matrix
        rhs : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            right hand side of matrix equation, representing the input or output
            vector
        """

        # convert matrix to full (non-sparse) format
        matrix = matrix.toarray()

        return EquationDisplay(matrix, rhs, self.elements)

    def circuit_matrix(self, matrix, rhs):
        """Get circuit matrix

        Parameters
        ----------
        matrix : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            left hand side of matrix equation, representing the circuit matrix
        rhs : :class:`~np.ndarray`, :class:`scipy.sparse.spmatrix`
            right hand side of matrix equation, representing the input or output
            vector
        """

        # convert matrix to full (non-sparse) format
        matrix = matrix.toarray()

        # column headers, with extra columns for component names and RHS
        headers = [""] + self.element_headers + ["RHS"]

        # create column vector of element names to go on left
        lhs = np.expand_dims(self.element_names, axis=1)

        return MatrixDisplay(lhs, matrix, rhs, headers)

class BaseEquation(object, metaclass=abc.ABCMeta):
    """Represents an equation.

    Parameters
    ----------
    coefficients : sequence of :class:`BaseCoefficient`
        Coefficients that make up the equation.
    """

    def __init__(self, coefficients):
        """Instantiate a new equation."""

        self.coefficients = []

        for coefficient in coefficients:
            self.add_coefficient(coefficient)

    def add_coefficient(self, coefficient):
        """Add coefficient to equation.

        Parameters
        ----------
        coefficient : :class:`BaseCoefficient`
            Coefficient to add.
        """

        self.coefficients.append(coefficient)

class ComponentEquation(BaseEquation):
    """Represents a component equation.

    Parameters
    ----------
    component : :class:`Component`
        Component associated with the equation.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new component equation."""

        # call parent constructor
        super(ComponentEquation, self).__init__(*args, **kwargs)

        self.component = component

class NodeEquation(BaseEquation):
    """Represents a node equation.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the equation.
    """

    def __init__(self, node, *args, **kwargs):
        """Instantiate a new node equation."""

        # call parent constructor
        super(NodeEquation, self).__init__(*args, **kwargs)

        self.node = node

class BaseCoefficient(object, metaclass=abc.ABCMeta):
    """Represents a coefficient.

    Parameters
    ----------
    value : :class:`float`
        Coefficient value.
    coefficient_type : {0, 1, 2}
        Coefficient type. Impedance is 0, current is 1, voltage is 2.
    """

    # coefficient types
    TYPE_IMPEDANCE = 0
    TYPE_CURRENT = 1
    TYPE_VOLTAGE = 2

    def __init__(self, value, coefficient_type):
        """Instantiate a new coefficient."""

        self.value = value
        self.coefficient_type = coefficient_type

    @property
    def coefficient_type(self):
        return self._coefficient_type

    @coefficient_type.setter
    def coefficient_type(self, coefficient_type):
        # validate coefficient
        if coefficient_type not in [self.TYPE_IMPEDANCE,
                                    self.TYPE_CURRENT,
                                    self.TYPE_VOLTAGE]:
            raise ValueError("Unrecognised coefficient type")

        self._coefficient_type = coefficient_type

class ImpedanceCoefficient(BaseCoefficient):
    """Represents an impedance coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this impedance coefficient represents.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new impedance coefficient."""

        self.component = component

        # call parent constructor
        super(ImpedanceCoefficient, self).__init__(
              coefficient_type=self.TYPE_IMPEDANCE, *args, **kwargs)

class CurrentCoefficient(BaseCoefficient):
    """Represents an current coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this current coefficient represents.
    """

    def __init__(self, component, *args, **kwargs):
        """Instantiate a new current coefficient."""

        self.component = component

        # call parent constructor
        super(CurrentCoefficient, self).__init__(
              coefficient_type=self.TYPE_CURRENT, *args, **kwargs)

class VoltageCoefficient(BaseCoefficient):
    """Represents a voltage coefficient.

    Parameters
    ----------
    node : :class:`Node`
        Node this voltage coefficient represents.
    """

    def __init__(self, node, *args, **kwargs):
        """Instantiate a new voltage coefficient."""

        self.node = node

        # call parent constructor
        super(VoltageCoefficient, self).__init__(
              coefficient_type=self.TYPE_VOLTAGE, *args, **kwargs)
