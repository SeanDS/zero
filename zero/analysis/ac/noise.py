import logging
import numpy as np

from .signal import AcSignalAnalysis
from ...data import NoiseDensity, Series

LOGGER = logging.getLogger(__name__)


class AcNoiseAnalysis(AcSignalAnalysis):
    """Small signal circuit analysis"""
    DEFAULT_INPUT_IMPEDANCE = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._noise_sink = None

    @property
    def noise_sink(self):
        return self._noise_sink

    @noise_sink.setter
    def noise_sink(self, sink):
        if not hasattr(sink, "name"):
            # This is an element name. Get the object. We use the user-supplied circuit here because
            # the copy may not have been created by this point.
            sink = self.circuit.get_element(sink)
        self._noise_sink = sink

    def calculate(self, input_type, sink, impedance=None, **kwargs):
        """Calculate noise from circuit elements at a particular element.

        Parameters
        ----------
        input_type : str
            Input type, either "voltage" or "current".
        sink : str or :class:`.Component` or :class:`.Node`
            The element to calculate noise at.
        impedance : float or :class:`.Quantity`, optional
            Input impedance. If None, the default is used.

        Returns
        -------
        :class:`~.solution.Solution`
            Solution containing noise spectra at the specified sink (or projected sink).
        """
        self.noise_sink = sink
        if impedance is None:
            LOGGER.warning(f"assuming default input impedance of {self.DEFAULT_INPUT_IMPEDANCE}")
            impedance = self.DEFAULT_INPUT_IMPEDANCE
        self._do_calculate(input_type, impedance=impedance, is_noise=True, **kwargs)

        return self.solution

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

    @property
    def right_hand_side_index(self):
        """Right hand side excitation component index"""
        return self.noise_element_index

    def _build_solution(self, noise_matrix):
        # empty noise sources
        empty = []

        # loop over circuit's noise sources
        for noise in self._current_circuit.noise_sources:
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
            self.solution.add_noise(NoiseDensity(source=noise, sink=self.noise_sink, series=series))

        if empty:
            empty_sources = ", ".join([str(response) for response in empty])
            LOGGER.debug(f"empty noise sources: {empty_sources}")

    def to_signal_analysis(self):
        """Return a new signal analysis using the settings defined in the current analysis."""
        return AcSignalAnalysis(self.circuit, print_progress=self.print_progress,
                                stream=self.stream)

    @property
    def noise_element_index(self):
        """Noise element matrix index"""
        try:
            return self.component_matrix_index(self.noise_sink)
        except ValueError:
            pass

        try:
            return self.node_matrix_index(self.noise_sink)
        except ValueError:
            pass

        raise ValueError(f"noise output element '{self.noise_sink}' is not in the circuit")
