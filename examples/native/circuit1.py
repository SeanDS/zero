"""A simple non-inverting whitening filter's response to a voltage input.

https://www.circuitlab.com/circuit/62vd4a/whitening-non-inverting/

Sean Leavey
"""

import numpy as np
from zero import Circuit
from zero.analysis import AcSignalAnalysis

if __name__ == "__main__":
    # 1000 frequencies between 1 Hz to 1 MHz.
    frequencies = np.logspace(0, 6, 1000)

    # Create circuit object.
    circuit = Circuit()

    # Add components.
    circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
    circuit.add_resistor(value="430", node1="n1", node2="nm")
    circuit.add_resistor(value="43k", node1="nm", node2="nout")
    circuit.add_capacitor(value="47p", node1="nm", node2="nout")
    circuit.add_library_opamp(name="op1", model="LT1124", node1="gnd", node2="nm", node3="nout")

    # Solve circuit.
    analysis = AcSignalAnalysis(circuit=circuit)
    solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1")

    # Plot.
    plot = solution.plot_responses(sinks=["nm", "nout", "op1"])
    plot.show()
