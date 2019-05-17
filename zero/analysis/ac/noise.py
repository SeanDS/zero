import logging
import numpy as np

from .base import BaseAcAnalysis
from ...data import NoiseDensity, Series

LOGGER = logging.getLogger(__name__)


class AcNoiseAnalysis(BaseAcAnalysis):
    """Small signal circuit analysis"""

    def __init__(self, element, **kwargs):
        # call parent constructor
        super().__init__(**kwargs)

        if not hasattr(element, "name"):
            # get element name from circuit
            element = self.circuit[element]

        self.element = element

    def validate_circuit(self):
        """Validate circuit for noise analysis"""
        pass

    def circuit_matrix(self, *args, **kwargs):
        """Calculate and return matrix used to solve for circuit noise at a \
        given frequency.

        Returns
        -------
        :class:`scipy.sparse.spmatrix`
            The circuit matrix.
        """

        # Return the transpose of the response matrix.
        return super().circuit_matrix(*args, **kwargs).T

    def right_hand_side(self):
        """Circuit noise (output) vector

        This creates a vector of size nx1, where n is the number of elements in
        the circuit, and sets the noise element's coefficient to 1 before
        returning it.

        Returns
        -------
        :class:`~np.ndarray`
            circuit's noise output vector
        """

        # create column vector
        e_n = self.get_empty_results_matrix(1)

        # set input to noise element
        e_n[self.noise_element_index, 0] = 1

        return e_n

    def calculate(self):
        """Calculate noise from circuit elements at a particular element.

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

            # multiply response from element to noise output element by noise entering
            # at that element, for all frequencies
            projected_noise = np.abs(response * spectral_density)

            # create series
            series = Series(x=self.frequencies, y=projected_noise)

            # add noise function to solution
            self.solution.add_noise(NoiseDensity(source=noise, sink=self.element, series=series))

        if empty:
            LOGGER.debug("empty noise sources: %s", ", ".join([str(response) for response in empty]))

    @property
    def noise_element_index(self):
        """Noise element matrix index"""
        try:
            return self.component_matrix_index(self.element)
        except ValueError:
            pass

        try:
            return self.node_matrix_index(self.element)
        except ValueError:
            pass

        raise ValueError(f"noise output element '{self.element}' is not in the circuit")

    @property
    def has_noise_input(self):
        """Check if circuit has a noise input."""
        return self.circuit.input_component.is_noise
