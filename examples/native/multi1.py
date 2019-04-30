"""Multiple circuits, same plot.

Sean Leavey
"""

import numpy as np
from zero import Circuit
from zero.analysis import AcSignalAnalysis


def circuit():
    circuit = Circuit()
    circuit.add_input(input_type="current", node="n1")
    circuit.add_capacitor(name="c1", value="10u", node1="gnd", node2="n1")
    circuit.add_resistor(value="430", node1="n1", node2="nm")
    circuit.add_resistor(value="43k", node1="nm", node2="nout")
    circuit.add_capacitor(value="47p", node1="nm", node2="nout")
    circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")
    return circuit

if __name__ == "__main__":
    frequencies = np.logspace(0, 6, 250)

    # Get circuits.
    circuit1 = circuit()
    circuit2 = circuit()

    # Change circuit2 value.
    circuit2["c1"].capacitance = "1u"

    # Solve circuits.
    analysis1 = AcSignalAnalysis(circuit=circuit1, frequencies=frequencies)
    analysis2 = AcSignalAnalysis(circuit=circuit2, frequencies=frequencies)
    analysis1.calculate()
    analysis2.calculate()

    # Combine solutions.
    solution = analysis1.solution + analysis2.solution

    # Plot
    solution.plot_tfs(sinks=["nout"])
    solution.show()
