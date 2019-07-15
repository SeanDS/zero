"""Base AC analysis tools"""

import sys
import abc
import logging
import statistics
from copy import copy
from collections import defaultdict
import numpy as np

from ..base import BaseAnalysis
from ...solve import DefaultSolver
from ...components import Component, Input, Node
from ...solution import Solution
from ...display import MatrixDisplay, EquationDisplay

LOGGER = logging.getLogger(__name__)


class BaseAcAnalysis(BaseAnalysis, metaclass=abc.ABCMeta):
    """Small signal circuit analysis"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create solver.
        self.solver = DefaultSolver()

        # Empty fields.
        self.frequencies = None
        self.input_type = None
        self._current_circuit = None
        self._solution = None
        self._node_sources = None
        self._node_sinks = None

    def reset(self):
        """Reset state of the analysis"""
        self.frequencies = None
        self.input_type = None
        self._current_circuit = None
        self._solution = None
        self._node_sources = None
        self._node_sinks = None

    def validate_circuit(self):
        """Validate circuit"""
        # Check input type.
        if self._current_circuit.input_component.input_type not in ["voltage", "current"]:
            raise ValueError("circuit input type must be either 'voltage' or 'current'")

    def component_index(self, component):
        """Get component serial number.

        Parameters
        ----------
        component : :class:`~.Component`
            component

        Returns
        -------
        :class:`int`
            component serial number

        Raises
        ------
        ValueError
            if component not found
        """
        return self._current_circuit.components.index(component)

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
            if ground node is specified or specified node is not found
        """
        if node == Node("gnd"):
            raise ValueError("ground node does not have an index")

        return list(self._current_circuit.non_gnd_nodes).index(node)

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
        yield from self._current_circuit.components
        yield from self._current_circuit.non_gnd_nodes

    @property
    def element_names(self):
        """Names of elements (components and nodes) within the circuit.

        Yields
        ------
        :class:`str`
            matrix element names
        """
        return [element.name for element in self.elements]

    @property
    def mean_resistance(self):
        """Average circuit resistance"""
        return statistics.mean([resistor.resistance
                                for resistor in self._current_circuit.resistors])

    @property
    def dim_size(self):
        """Circuit matrix dimension size

        Returns
        -------
        :class:`int`
            number of rows/columns in circuit matrix
        """

        # Dimension size is the number of components added to circuit, including the input, plus the
        # number of non-ground nodes.
        return self._current_circuit.n_components + len(list(self._current_circuit.non_gnd_nodes))

    @property
    def n_freqs(self):
        return len(self.frequencies)

    @property
    def solution(self):
        if self._solution is None:
            self._solution = Solution(self.frequencies)

        return self._solution

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

    @abc.abstractmethod
    def calculate(self):
        """Calculate solution."""
        raise NotImplementedError

    def _do_calculate(self, input_type, frequencies, print_equations=False, print_matrix=False,
                      **inputs):
        """Calculate analysis results."""
        # Reset state.
        self.reset()

        self.frequencies = np.array(frequencies)

        # Make a copy of the circuit. This allows us to call calculate(), which adds an input
        # component, multiple times.
        self._current_circuit = copy(self.circuit)
        # Add input.
        self._set_input(input_type, **inputs)
        # Validate.
        self.validate_circuit()

        if print_equations:
            print(self.circuit_equation_display(), file=self.stream)

        if print_matrix:
            print(self.circuit_matrix_display(), file=self.stream)

        # Calculate transfer functions by solving the transfer matrix for input at the circuit's
        # input node/component.
        responses = self.solve()

        self._build_solution(responses)

    def _set_input(self, input_type, impedance=None, is_noise=False, node=None, node_p=None,
                   node_n=None):
        """Set circuit input.

        Supports either a single node (for grounded voltage input) or a pair of nodes (for floating
        voltage inputs).
        """
        # Handle nodes.
        if node is not None:
            if node_p is not None or node_n is not None:
                raise ValueError("node cannot be specified alongside node_p or node_n")

            node_n = Node("gnd")
            node_p = node
        else:
            if node_p is None and node_n is None:
                # No nodes specified.
                raise ValueError("input node(s) must be specified")
            elif node_p is None or node_n is None:
                # Only one of node_p or node_n specified.
                raise ValueError("node_p and node_n must both be specified")

        input_type = input_type.lower()

        if input_type == "voltage":
            self.input_type = "voltage"
        elif input_type == "current":
            self.input_type = "current"
        else:
            raise ValueError("unrecognised input type")

        # Create input component.
        self._create_input_component(node_n, node_p, impedance, is_noise)

    def _create_input_component(self, node_n, node_p, impedance, is_noise):
        """Create circuit input component."""
        if self._current_circuit.has_input:
            raise Exception("circuit already has input")

        self._current_circuit.add_component(Input([node_n, node_p], self.input_type,
                                                  impedance=impedance, is_noise=is_noise))

    @abc.abstractmethod
    def _build_solution(self, results_matrix):
        """Build solution with the given results matrix."""
        raise NotImplementedError

    def right_hand_side(self):
        """Circuit signal excitation vector.

        This creates a vector of size nx1, where n is the number of elements in the circuit, with
        all elements zero except for the excitation component, which is set to 1.

        Returns
        -------
        :class:`np.ndarray`
            The circuit's excitation vector.
        """
        # Create column vector.
        y = self.get_empty_results_matrix(1)

        # Set input to input component.
        y[self.right_hand_side_index, 0] = 1

        return y

    @property
    @abc.abstractmethod
    def right_hand_side_index(self):
        """Right hand side excitation component index"""
        raise NotImplementedError

    def circuit_matrix(self, frequency):
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

        # add sources and sinks
        self.set_up_sources_and_sinks()

        # Kirchoff's voltage law / op-amp voltage gain equations
        for equation in self.component_equations:
            for coefficient in equation.coefficients:
                # default row index
                row = self.component_matrix_index(equation.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                if coefficient.TYPE == "impedance":
                    # use target component column
                    column = self.component_matrix_index(coefficient.component)
                elif coefficient.TYPE == "voltage":
                    # includes extra I[in] column (at index == self.n_components)
                    column = self.node_matrix_index(coefficient.node)
                else:
                    raise ValueError("invalid coefficient type")

                matrix[row, column] = value

        # Kirchoff's current law
        for equation in self.node_equations:
            for coefficient in equation.coefficients:
                if not coefficient.TYPE == "current":
                    raise ValueError("invalid coefficient type")

                row = self.node_matrix_index(equation.node)
                column = self.component_matrix_index(coefficient.component)

                if callable(coefficient.value):
                    value = coefficient.value(frequency)
                else:
                    value = coefficient.value

                matrix[row, column] = value

        return matrix

    def solve(self):
        """Solve the circuit.

        Solves matrix equation Ax = b, where A is the circuit matrix and b is the right hand side.

        Returns
        -------
        :class:`~np.ndarray`
            The inverse of the circuit matrix.
        """
        # results matrix
        results = self.get_empty_results_matrix(self.n_freqs)

        # update progress every 1% of the way there
        update = self.n_freqs // 100

        # if there are less than 100 frequencies, update progress after
        # every frequency
        if update == 0:
            update += 1

        # create frequency generator with progress bar
        freq_gen = self.progress(self.frequencies, self.n_freqs, update=update)

        # right hand side to solve against
        rhs = self.right_hand_side()

        # frequency loop
        for index, frequency in enumerate(freq_gen):
            # get matrix for this frequency, converting to CSR format for
            # efficient solving
            matrix = self.circuit_matrix(frequency).tocsr()

            # call solver function
            results[:, index] = self.solver.solve(matrix, rhs)

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
        siso_components = list(self._current_circuit.passive_components)
        siso_components += [self._current_circuit.input_component]

        # single-input, single-output components sink to their first node and source from
        # their second
        for component in siso_components:
            if component.node1:
                self._node_sinks[component.node1].add(component) # current flows into here...
            if component.node2:
                self._node_sources[component.node2].add(component) # ...and out of here

        # op-amps source current from their third (output) node (their input nodes are
        # ideal and therefore don't source or sink current)
        for component in self._current_circuit.opamps:
            self._node_sources[component.node3].add(component) # current flows out of here

    def component_equation(self, component):
        """Equation representing circuit component

        The component equation represents the component in the circuit matrix. It maps
        input or noise sources to voltages across :class:`nodes <Node>` and currents
        through :class:`components <Component>` in terms of their
        :class:`impedance <ImpedanceCoefficient>` and
        :class:`voltage <VoltageCoefficient>` coefficients.

        Note that special behaviour is applied to op-amp voltage outputs if they are
        configured as voltage followers.

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
            if (component.input_type == "current"
                or (hasattr(component, "is_noise") and component.is_noise)):
                # set source impedance
                coefficients.append(ImpedanceCoefficient(component=component,
                                                         value=component.impedance))

            if (component.input_type == "voltage"
                or (hasattr(component, "is_noise") and component.is_noise)):
                # set sink node coefficient
                if component.node_n is not Node("gnd"):
                    # voltage
                    coefficients.append(VoltageCoefficient(node=component.node_n, value=-1))

                # set source node coefficient
                if component.node_p is not Node("gnd"):
                    # voltage
                    coefficients.append(VoltageCoefficient(node=component.node_p, value=1))
        else:
            # add input node coefficient
            if component.node1 is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=component.node1, value=-1))

            # add output node coefficient
            if component.node2 is not Node("gnd"):
                # voltage
                coefficients.append(VoltageCoefficient(node=component.node2, value=1))

            if hasattr(component, "gain"):
                # this is an op-amp
                if component.node3 == component.node2:
                    # this is a voltage follower
                    inverse_gain = lambda f: 1 + component.inverse_gain(f)
                else:
                    # add voltage gain V[n3] = H(s) (V[n1] - V[n2])
                    inverse_gain = component.inverse_gain

                coefficients.append(VoltageCoefficient(node=component.node3,
                                                       value=inverse_gain))
            else:
                # this is a passive component
                # add impedance
                coefficients.append(ImpedanceCoefficient(component=component,
                                                         value=component.impedance))

                # add mutual inductances, if present
                if hasattr(component, "coupled_inductors"):
                    for inductor in component.coupled_inductors:
                        # impedance created by the coupled inductor
                        coupled_impedance = lambda f, i=inductor: component.impedance_from(i, f)

                        # create coefficient containing impedance from other inductor
                        coefficients.append(ImpedanceCoefficient(component=inductor,
                                                                 value=coupled_impedance))

        # create and return equation
        return ComponentEquation(component, coefficients=coefficients)

    def node_equation(self, node):
        """Equation representing circuit node

        This should be called after :meth:`set_up_sources_and_sinks`.
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

        return [self.component_equation(component) for component in self._current_circuit.components]

    @property
    def node_equations(self):
        """Linear equations representing circuit nodes

        Yields
        ------
        :class:`~.components.NodeEquation`
            sequence of node equations
        """

        return [self.node_equation(node) for node in self._current_circuit.non_gnd_nodes]

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

        return self._current_circuit.n_components + self.node_index(node)

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
            return f"i[{element.name}]"
        if isinstance(element, Node):
            return f"V[{element.name}]"

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

    def circuit_equation_display(self, frequency=1):
        """Get circuit equations

        Parameters
        ----------
        frequency : :class:`float`, optional
            Frequency to display circuit equations for.
        """

        matrix = self.circuit_matrix(frequency=frequency)

        # convert matrix to full (non-sparse) format
        matrix = matrix.toarray()

        return EquationDisplay(matrix, self.right_hand_side(), self.elements)

    def circuit_matrix_display(self, frequency=1):
        """Get circuit matrix

        Parameters
        ----------
        frequency : :class:`float`, optional
            Frequency to display circuit equations for.
        """

        matrix = self.circuit_matrix(frequency=frequency)

        # convert matrix to full (non-sparse) format
        matrix = matrix.toarray()

        # column headers, with extra columns for component names and RHS
        headers = [""] + self.element_headers + ["RHS"]

        # create column vector of element names to go on left
        lhs = np.expand_dims(self.element_names, axis=1)

        return MatrixDisplay(lhs, matrix, self.right_hand_side(), headers)


class BaseEquation(metaclass=abc.ABCMeta):
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

    def __init__(self, component, **kwargs):
        """Instantiate a new component equation."""

        # call parent constructor
        super().__init__(**kwargs)

        self.component = component


class NodeEquation(BaseEquation):
    """Represents a node equation.

    Parameters
    ----------
    node : :class:`Node`
        Node associated with the equation.
    """

    def __init__(self, node, **kwargs):
        """Instantiate a new node equation."""

        # call parent constructor
        super().__init__(**kwargs)

        self.node = node


class BaseCoefficient(metaclass=abc.ABCMeta):
    """Represents a coefficient.

    Parameters
    ----------
    value : :class:`float`
        Coefficient value.
    """

    TYPE = ""

    def __init__(self, value):
        """Instantiate a new coefficient."""
        self.value = value


class ComponentCoefficient(BaseCoefficient):
    """Represents a component coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this coefficient represents.
    """
    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        self.component = component


class ImpedanceCoefficient(ComponentCoefficient):
    """Represents an impedance coefficient."""
    TYPE = "impedance"


class CurrentCoefficient(ComponentCoefficient):
    """Represents an current coefficient."""
    TYPE = "current"


class VoltageCoefficient(BaseCoefficient):
    """Represents a voltage coefficient.

    Parameters
    ----------
    node : :class:`Node`
        Node this voltage coefficient represents.
    """

    TYPE = "voltage"

    def __init__(self, node, **kwargs):
        self.node = node

        # call parent constructor
        super().__init__(**kwargs)
