import logging
import numpy as np

from .signal import AcSignalAnalysis
from ...data import NoiseDensity, MultiNoiseDensity, Series

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

    def calculate(self, input_type, sink, impedance=None, incoherent_sum=False, input_refer=False,
                  **kwargs):
        """Calculate noise from circuit elements at a particular element.

        Parameters
        ----------
        input_type : str
            Input type, either "voltage" or "current".
        sink : str or :class:`.Component` or :class:`.Node`
            The element to calculate noise at.
        impedance : float or :class:`.Quantity`, optional
            Input impedance. If None, the default is used.
        incoherent_sum : :class:`bool` or :class:`dict`, optional
            Incoherent sum specification. If True, the incoherent sum of all noise in the circuit at
            the sink is calculated and added to the solution. Alternatively, this parameter can be
            specified as a dict containing labels as keys and sequences of noise sources as values.
            The noise sources can be either :class:`.NoiseDensity` objects or noise specifier
            strings as supported by :meth:`.Solution.get_noise`. The values may alternatively be the
            strings "all", "allop" or "allr" to compute noise from all components, all op-amps and
            all resistors, respectively. Sums are plotted in shades of grey determined by the
            plotting configuration's ``sum_greyscale_cycle_start``, ``sum_greyscale_cycle_stop`` and
            ``sum_greyscale_cycle_count`` values.
        input_refer : bool, optional
            Refer the noise to the input.

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
        self.noise_sink = sink
        if impedance is None:
            LOGGER.warning(f"assuming default input impedance of {self.DEFAULT_INPUT_IMPEDANCE}")
            impedance = self.DEFAULT_INPUT_IMPEDANCE
        self._do_calculate(input_type, impedance=impedance, is_noise=True, **kwargs)
        if incoherent_sum:
            self._compute_sums(incoherent_sum)
        if input_refer:
            self._refer_sink_noise_to_input()
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

            if noise.element_type == "component":
                # noise is from a component; use its matrix index
                index = self.component_matrix_index(noise.component)
            elif noise.element_type == "node":
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

    def _compute_sums(self, sum_spec):
        """Compute incoherent noise sums and add them to the solution.

        Parameters
        ----------
        sum_spec : :class:`bool` or :class:`dict`
            Incoherent sum specification. If True, the incoherent sum of all noise in the circuit at
            the sink is calculated and added to the solution. Alternatively, this parameter can be
            specified as a dict containing labels as keys and sequences of noise sources as values.
            The noise sources can be either :class:`.NoiseDensity` objects or noise specifier
            strings as supported by :meth:`.Solution.get_noise`. The values may alternatively be the
            strings "all", "allop" or "allr" to compute noise from all components, all op-amps and
            all resistors, respectively. Sums are plotted in shades of grey determined by the
            plotting configuration's ``sum_greyscale_cycle_start``, ``sum_greyscale_cycle_stop`` and
            ``sum_greyscale_cycle_count`` values.
        """
        if sum_spec is True:
            # Sum using all noise and the default MultiNoiseDensity label.
            sum_spec = {None: self.solution.noise[self.solution.DEFAULT_GROUP_NAME]}
        for label, spectra in sum_spec.items():
            if spectra is None:
                raise ValueError("noise sum spectra cannot be empty")
            if isinstance(spectra, str):
                identifier = spectra.lower()
                if identifier == "all":
                    constituents = self.solution.noise[self.solution.DEFAULT_GROUP_NAME]
                elif identifier == "allop":
                    constituents = self.solution.opamp_noise[self.solution.DEFAULT_GROUP_NAME]
                elif identifier == "allr":
                    constituents = self.solution.resistor_noise[self.solution.DEFAULT_GROUP_NAME]
                else:
                    raise ValueError(f"unrecognised noise collection '{spectra}'")
            else:
                constituents = []
                for spectrum in spectra:
                    if not isinstance(spectrum, NoiseDensity):
                        spectrum = self.solution.get_noise(source=spectrum, sink=self.noise_sink)
                    constituents.append(spectrum)

            self.solution.add_noise_sum(MultiNoiseDensity(constituents=constituents,
                                                          sink=self.noise_sink, label=label))

    def _refer_sink_noise_to_input(self):
        """Project the calculated noise to the input."""
        LOGGER.info("projecting noise to input")

        input_component = self._current_circuit.input_component
        if self.input_type == "voltage":
            input_element = input_component.node2
        else:
            input_element = input_component
        projection_analysis = self.to_signal_analysis()
        # Grab the input nodes from the noise circuit.
        node_n, node_p = input_component.nodes
        projection = projection_analysis.calculate(frequencies=self.frequencies,
                                                   input_type=self.input_type, node_n=node_n,
                                                   node_p=node_p)
        # Transfer function from input to noise sink.
        input_response = projection.get_response(source=input_element, sink=self.noise_sink)

        for __, noise_spectra in self.solution.noise.items():
            for noise in noise_spectra:
                self.solution.replace(noise, noise * input_response.inverse())

        for __, noise_sums in self.solution.noise_sums.items():
            for noise in noise_sums:
                self.solution.replace(noise, noise * input_response.inverse())

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
