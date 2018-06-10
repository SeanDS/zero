import abc
import sys
import logging
import numpy as np
from collections import defaultdict

from ..base import BaseAnalysis
from ...solve import DefaultSolver
from ...components import Component, Node
from ...solution import Solution
from ...display import MatrixDisplay, EquationDisplay

LOGGER = logging.getLogger("ac-analysis")

class BaseAcAnalysis(BaseAnalysis):
    """Small signal circuit analysis"""

    def __init__(self, frequencies, prescale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frequencies = np.array(frequencies)
        self.prescale = bool(prescale)

        # create solver
        self.solver = DefaultSolver()

        # empty fields
        self._solution = None

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

    @property
    def n_freqs(self):
        return len(self.frequencies)

    @property
    def solution(self):
        if self._solution is None:
            self._solution = Solution(self.circuit, self.frequencies)
        
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

    @abc.abstractmethod
    def right_hand_side(self):
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

                if coefficient.TYPE == "impedance":
                    # don't change indices, but scale impedance
                    value *= scale
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
        super().__init__(*args, **kwargs)

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
        super().__init__(*args, **kwargs)

        self.node = node

class BaseCoefficient(object, metaclass=abc.ABCMeta):
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

class ImpedanceCoefficient(BaseCoefficient):
    """Represents an impedance coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this impedance coefficient represents.
    """

    TYPE = "impedance"

    def __init__(self, component, *args, **kwargs):
        self.component = component

        # call parent constructor
        super().__init__(*args, **kwargs)

class CurrentCoefficient(BaseCoefficient):
    """Represents an current coefficient.

    Parameters
    ----------
    component : :class:`Component`
        Component this current coefficient represents.
    """

    TYPE = "current"

    def __init__(self, component, *args, **kwargs):
        self.component = component

        # call parent constructor
        super().__init__(*args, **kwargs)

class VoltageCoefficient(BaseCoefficient):
    """Represents a voltage coefficient.

    Parameters
    ----------
    node : :class:`Node`
        Node this voltage coefficient represents.
    """

    TYPE = "voltage"

    def __init__(self, node, *args, **kwargs):
        self.node = node

        # call parent constructor
        super().__init__(*args, **kwargs)
