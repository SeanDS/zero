"""Native circuit construction and simulation

This simulates a simple non-inverting whitening filter's noise at its output.

https://www.circuitlab.com/circuit/62vd4a/whitening-non-inverting/

Sean Leavey
"""

import numpy as np
from zero import Circuit
from zero.analysis import AcNoiseAnalysis

if __name__ == "__main__":
    # 1000 frequencies between 1 Hz to 1 MHz
    frequencies = np.logspace(0, 6, 1000)

    # create circuit object
    circuit = Circuit()

    # add components
    circuit.add_input(input_type="noise", node="n1", impedance=50)
    circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
    circuit.add_resistor(name="r1", value="430", node1="n1", node2="nm")
    circuit.add_resistor(value="43k", node1="nm", node2="nout")
    circuit.add_capacitor(value="47p", node1="nm", node2="nout")
    circuit.add_library_opamp(name="o1", model="LT1124", node1="gnd", node2="nm", node3="nout")

    # solve circuit
    analysis = AcNoiseAnalysis(node="nout", circuit=circuit, frequencies=frequencies)

    analysis.calculate()
    solution = analysis.solution

    # plot, with sums of resistors and op-amp noise sources
    solution.plot_noise(sources=["R(r1)", "I(o1, nm)"],
                        compute_sum_sources={"all resistors": circuit.resistor_noise_sources,
                                            "all op-amps": circuit.opamp_noise_sources})
    solution.show()
