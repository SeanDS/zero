import logging
import numpy as np

from .base import BaseAcAnalysis
from ...components import Node
from ...data import NoiseSpectrum, Series

LOGGER = logging.getLogger("ac-analysis")

class AcNoiseAnalysis(BaseAcAnalysis):
    """Small signal circuit analysis"""

    def __init__(self, node, **kwargs):
        # empty fields
        self._node = None

        self.node = node
    
        # call parent constructor
        super().__init__(**kwargs)

    def validate_circuit(self):
        """Validate circuit for noise analysis"""
        # check input
        if self.circuit.input_component.input_type != "noise":
            raise ValueError("circuit input type must be 'noise'")

    @property
    def node(self):
        """Circuit noise node

        Returns
        -------
        :class:`~.components.Node`
            The circuit's noise node
        """
        return self._node

    @node.setter
    def node(self, node):
        """Set circuit's noise node

        Parameters
        ----------
        node : :class:`~.components.Node`
            The circuit's new noise node
        """

        if not isinstance(node, Node):
            node = Node(node)

        self._node = node

    def circuit_matrix(self, *args, **kwargs):
        """Calculate and return matrix used to solve for circuit noise at a \
        given frequency.

        Returns
        -------
        :class:`scipy.sparse.spmatrix`
            The circuit matrix.
        """

        # return the transpose of the transfer function matrix
        return super().circuit_matrix(*args, **kwargs).T

    def right_hand_side(self):
        """Circuit noise (output) vector

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

    def calculate(self):
        """Calculate noise from circuit :class:`component <.Component>` / \
        :class:`node <.Node>` at a particular :class:`node <.Node>`.

        Returns
        -------
        :class:`~.solution.Solution`
            solution

        Raises
        ------
        Exception
            If no input is present within the circuit.
        Exception
            If no noise sources are defined.
        """

        if not self.circuit.has_input:
            raise Exception("circuit must contain an input")

        # calculate noise functions by solving the transfer matrix for input
        # at the circuit's noise sources
        noise_matrix = self.solve()

        # scale vector, for converting units, if necessary
        scale = self.get_empty_results_matrix(1)
        scale[:, 0] = 1

        if self.prescale:
            # convert currents from natural units back to amperes
            prescaler = 1 / self.mean_resistance

            for node in self.circuit.non_gnd_nodes:
                scale[self.node_matrix_index(node), 0] = 1 / prescaler

        # unscale
        noise_matrix *= scale

        self._build_solution(noise_matrix)

    def _build_solution(self, noise_matrix):
        # empty noise sources
        empty = []

        # loop over circuit's noise sources
        for noise in self.circuit.noise_sources:
            # get this element's noise spectral density
            spectral_density = noise.spectral_density(frequencies=self.frequencies)

            if np.all(spectral_density) == 0:
                # null noise source
                empty.append(noise)

            if noise.TYPE == "component":
                # noise is from a component; use its matrix index
                index = self.component_matrix_index(noise.component)
            elif noise.TYPE == "node":
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
            series = Series(x=self.frequencies, y=projected_noise)

            # add noise function to solution
            self.solution.add_noise(NoiseSpectrum(source=noise, sink=self.node,
                                                  series=series))

        if len(empty):
            LOGGER.debug("empty noise sources: %s", ", ".join([str(tf) for tf in empty]))

    @property
    def noise_node_index(self):
        """Noise node matrix index"""
        return self.node_matrix_index(self.node)

    @property
    def has_noise_input(self):
        """Check if circuit has a noise input."""
        return self.circuit.input_component.input_type == "noise"