import logging
import numpy as np

from .base import BaseAcAnalysis
from ...data import Response, Series
from ...components import Component, Node

LOGGER = logging.getLogger(__name__)


class AcSignalAnalysis(BaseAcAnalysis):
    """AC signal analysis"""
    def validate_circuit(self):
        """Validate circuit for signal analysis"""
        # check input
        if self.circuit.input_component.input_type not in ["voltage", "current"]:
            raise ValueError("circuit input type must be either 'voltage' or 'current'")

    def right_hand_side(self):
        """Circuit signal (input) vector.

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

    def calculate(self):
        """Calculate circuit response from input component or node to output component or node.

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
        """

        if not self.circuit.has_input:
            raise Exception("circuit must contain an input")

        # Calculate responses by solving the transfer matrix for input at the circuit's input
        # node/component.
        responses = self.solve()

        self._build_solution(responses)

    def _build_solution(self, responses):
        # Empty responses.
        empty = []

        # Output component indices.
        for component in self.circuit.components:
            # Extract response for this component.
            response = responses[self.component_matrix_index(component), :]

            if np.all(response) == 0:
                # Null response.
                empty.append(component)

            # Create data series.
            series = Series(x=self.frequencies, y=response)

            # Create appropriate response function depending on input type.
            if self.has_voltage_input:
                source = self.circuit.input_component.node_p
            elif self.has_current_input:
                source = self.circuit.input_component
            else:
                raise ValueError("specify either a current or voltage input")

            function = Response(source=source, sink=component, series=series)

            # Add response to solution.
            self.solution.add_response(function)

        # Output node indices.
        for node in self.circuit.non_gnd_nodes:
            # Extract response for this node.
            response = responses[self.node_matrix_index(node), :]

            if np.all(response) == 0:
                # Null response.
                empty.append(node)

            # Create series.
            series = Series(x=self.frequencies, y=response)

            # Create appropriate response function depending on input type.
            if self.has_voltage_input:
                source = self.circuit.input_component.node_p
            elif self.has_current_input:
                source = self.circuit.input_component
            else:
                raise ValueError("specify either a current or voltage input")

            function = Response(source=source, sink=node, series=series)

            # Add response to solution.
            self.solution.add_response(function)

        if len(empty):
            LOGGER.debug("empty responses: %s", ", ".join([str(response) for response in empty]))

    @property
    def input_component_index(self):
        """Input component's matrix index"""
        return self.component_matrix_index(self.circuit.input_component)

    @property
    def input_node_index(self):
        """Input node's matrix index"""
        return self.node_matrix_index(self.circuit.input_component.node2)

    @property
    def has_voltage_input(self):
        """Check if circuit has a voltage input."""
        return self.circuit.input_component.input_type == "voltage"

    @property
    def has_current_input(self):
        """Check if circuit has a current input."""
        return self.circuit.input_component.input_type == "current"
