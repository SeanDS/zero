import logging
import numpy as np

from .base import BaseAcAnalysis
from ...data import (VoltageVoltageTF, VoltageCurrentTF, CurrentCurrentTF,
                     CurrentVoltageTF, Series)
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
        """Calculate circuit transfer functions from input \
        :class:`component <.Component>` / :class:`node <.Node>` to output \
        :class:`components <.Component>` / :class:`nodes <.Node>`.

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

        # calculate transfer functions by solving the transfer matrix for input
        # at the circuit's input node/component
        tfs = self.solve()

        # scale vector, for converting units, if necessary
        scale = self.get_empty_results_matrix(1)
        scale[:, 0] = 1

        if self.prescale:
            # convert currents from natural units back to amperes
            prescaler = 1 / self.mean_resistance

            for component in self.circuit.components:
                scale[self.component_matrix_index(component), 0] = prescaler

        # unscale
        tfs *= scale

        self._build_solution(tfs)

    def _build_solution(self, tfs):
        # empty tfs
        empty = []

        # output component indices
        for component in self.circuit.components:
            # extract transfer function for this component
            tf = tfs[self.component_matrix_index(component), :]

            if np.all(tf) == 0:
                # null transfer function
                empty.append(component)

            # create data series
            series = Series(x=self.frequencies, y=tf)

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
            self.solution.add_tf(function)

        # output node indices
        for node in self.circuit.non_gnd_nodes:
            # extract transfer function for this node
            tf = tfs[self.node_matrix_index(node), :]

            if np.all(tf) == 0:
                # null transfer function
                empty.append(node)

            # create series
            series = Series(x=self.frequencies, y=tf)

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
            self.solution.add_tf(function)

        if len(empty):
            LOGGER.debug("empty transfer functions: %s", ", ".join([str(tf) for tf in empty]))

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
