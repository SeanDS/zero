
"""Documentation example. Requires target figure filename as argument."""

import sys
import numpy as np
from zero import Circuit
from zero.analysis import AcNoiseAnalysis
from zero.noise import VoltageNoise

# Create a new noise type.
class ResistorCurrentNoise(VoltageNoise):
    """Resistor current noise source.

    This models resistor current noise. See e.g. https://dcc.ligo.org/LIGO-T0900200/public
    for more details. This noise depends on resistor composition and on its current. Be
    careful when using this noise - it generally does not transfer to different circuits
    with identical resistors as it depends on the voltage drop across the resistor.

    Parameters
    ----------
    vnoise : :class:`float`
        The voltage noise at the specified frequency (V/sqrt(Hz)).
    frequency : :class:`float`
        The frequency at which the specified voltage noise is defined (Hz).
    exponent : :class:`float`
        The frequency exponent to use for calculating the frequency response.
    """
    def __init__(self, vnoise, frequency=1.0, exponent=0.5, **kwargs):
        super().__init__(**kwargs)
        self.vnoise = vnoise
        self.frequency = frequency
        self.exponent = exponent

    def noise_voltage(self, frequencies, **kwargs):
        return self.vnoise * self.frequency / frequencies ** self.exponent

    @property
    def label(self):
        return f"RE({self.component.name})"


# 1000 frequencies between 0.1 Hz to 10 kHz
frequencies = np.logspace(-1, 4, 1000)

# Create circuit object.
circuit = Circuit()

# Add components.
circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
circuit.add_resistor(value="430", node1="n1", node2="nm", name="r1")
circuit.add_resistor(value="43k", node1="nm", node2="nout")
circuit.add_capacitor(value="47p", node1="nm", node2="nout")
circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")

# Add resistor current noise to r1 with 10 nV/sqrt(Hz) at 1 Hz, with 1/f^2 drop-off.
r1 = circuit["r1"]
r1.add_noise(ResistorCurrentNoise(vnoise=1e-8, frequency=1.0, exponent=0.5))

# Solve circuit.
analysis = AcNoiseAnalysis(circuit=circuit)
solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1",
                                sink="nout", incoherent_sum=True)

# Plot.
plotter = solution.plot_noise(sink="nout")
plotter.save(sys.argv[1])
