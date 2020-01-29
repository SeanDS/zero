"""Photodetector readout circuit noise, projected to the circuit's input, projected into
displacement noise.

Rana Adhikari, Sean Leavey
"""

import numpy as np
from zero import Circuit
from zero.analysis import AcNoiseAnalysis
from zero.tools import create_response

if __name__ == "__main__":
    # 1000 frequencies between 0.1 Hz to 100 kHz
    frequencies = np.logspace(-1, 5, 1000)

    # Create circuit object.
    circuit = Circuit()

    # The photodiode is a current source that connects through a photodiode circuit model (shunt
    # capacitor and series resistor).
    circuit.add_capacitor(value="200p", node1="gnd", node2="nD")
    circuit.add_resistor(value="10", node1="nD", node2="nm")

    # Transimpedance amplifier.
    circuit.add_library_opamp(model="OP27", node1="gnd", node2="nm", node3="nout")
    circuit.add_resistor(value="1k", node1="nm", node2="nout")

    # Solve circuit.
    analysis = AcNoiseAnalysis(circuit=circuit)
    solution = analysis.calculate(frequencies=frequencies, input_type="current", node="nD",
                                  sink="nout", impedance="1G", incoherent_sum=True,
                                  input_refer=True)

    # Scale all noise at the input to displacement.
    pd_to_displacement = create_response(source="input", sink="displacement", source_unit="A",
                                         sink_unit="m", data=1e-9*np.ones_like(frequencies),
                                         frequencies=frequencies)
    solution.scale_noise(pd_to_displacement, sink="input")

    # Plot. Note that the sink is now the input, since we projected the noise there.
    plot = solution.plot_noise(sink="displacement", ylim=(1e-23, 1e-18),
                               title="Photodetector noise")
    plot.show()
