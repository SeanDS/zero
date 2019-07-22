import logging
import numpy as np

from .base import BaseAcAnalysis
from ...data import Response, Series

LOGGER = logging.getLogger(__name__)


class AcSignalAnalysis(BaseAcAnalysis):
    """AC signal analysis"""
    def calculate(self, input_type, **kwargs):
        """Calculate responses.

        Parameters
        ----------
        input_type : str
            Input type, either "voltage" or "current".

        Other Parameters
        ----------------
        frequencies : :class:`np.ndarray` or sequence
            The frequency vector to calculate the response with.
        node, node_p, node_n : :class:`.Node`
            The node or nodes to make the input. The `node` parameter sets a single, grounded input,
            whereas `node_p` and `node_n` together create a floating input.
        print_equations : :class:`bool`, optional
            Print the circuit equations.
        print_matrix : :class:`bool`, optional
            Print the circuit matrix.

        Returns
        -------
        :class:`~.solution.Solution`
            Solution containing noise spectra at the specified sink (or projected sink).
        """
        if input_type == "current":
            # Set impedance to give correct scaling.
            impedance = 1
        else:
            impedance = None
        self._do_calculate(input_type, impedance=impedance, **kwargs)
        return self.solution

    @property
    def right_hand_side_index(self):
        """Right hand side excitation component index"""
        return self.input_component_index

    def _build_solution(self, responses):
        # Empty responses.
        empty = []

        # Output component indices.
        for component in self._current_circuit.components:
            # Extract response for this component.
            response = responses[self.component_matrix_index(component), :]

            if np.all(response) == 0:
                # Null response.
                empty.append(component)

            # Create data series.
            series = Series(x=self.frequencies, y=response)

            # Create appropriate response function depending on input type.
            if self.has_voltage_input:
                source = self._current_circuit.input_component.node_p
            elif self.has_current_input:
                source = self._current_circuit.input_component
            else:
                raise ValueError("specify either a current or voltage input")

            function = Response(source=source, sink=component, series=series)

            # Add response to solution.
            self.solution.add_response(function)

        # Output node indices.
        for node in self._current_circuit.non_gnd_nodes:
            # Extract response for this node.
            response = responses[self.node_matrix_index(node), :]

            if np.all(response) == 0:
                # Null response.
                empty.append(node)

            # Create series.
            series = Series(x=self.frequencies, y=response)

            # Create appropriate response function depending on input type.
            if self.has_voltage_input:
                source = self._current_circuit.input_component.node_p
            elif self.has_current_input:
                source = self._current_circuit.input_component
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
        return self.component_matrix_index(self._current_circuit.input_component)

    @property
    def input_node_index(self):
        """Input node's matrix index"""
        return self.node_matrix_index(self._current_circuit.input_component.node2)

    @property
    def has_voltage_input(self):
        """Check if circuit has a voltage input."""
        return self._current_circuit.input_component.input_type == "voltage"

    @property
    def has_current_input(self):
        """Check if circuit has a current input."""
        return self._current_circuit.input_component.input_type == "current"
