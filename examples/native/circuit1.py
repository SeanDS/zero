"""Native circuit construction and simulation

This simulates a simple non-inverting whitening filter's transfer functions for
a voltage input.

https://www.circuitlab.com/circuit/62vd4a/whitening-non-inverting/

Sean Leavey
"""

import numpy as np
from circuit import Circuit
from circuit.analysis import AcSignalAnalysis

# 1000 frequencies between 1 Hz to 1 MHz
frequencies = np.logspace(0, 6, 1000)

# create circuit object
circuit = Circuit()

# add components
circuit.add_input(input_type="voltage", node="n1")
circuit.add_capacitor(name="c1", value="10u", node1="gnd", node2="n1")
circuit.add_resistor(name="r1", value="430", node1="n1", node2="nm")
circuit.add_resistor(name="r2", value="43k", node1="nm", node2="nout")
circuit.add_capacitor(name="c2", value="47p", node1="nm", node2="nout")
circuit.add_library_opamp(name="o1", model="LT1124", node1="gnd", node2="nm",
                          node3="nout")

# solve circuit
analysis = AcSignalAnalysis(circuit=circuit, frequencies=frequencies)

analysis.calculate()
solution = analysis.solution

# plot
solution.plot_tfs(sinks=["nm", "nout", "r1", "r2", "o1"])
solution.show()