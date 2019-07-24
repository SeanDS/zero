"""Plotting multiple analysis results together.

Sean Leavey
"""

import numpy as np
from zero import Circuit
from zero.analysis import AcSignalAnalysis


def circuit():
    circuit = Circuit()
    circuit.add_resistor(value="430", node1="n1", node2="nm")
    circuit.add_capacitor(name="c1", value="10u", node1="nm", node2="gnd")
    circuit.add_resistor(value="43k", node1="nm", node2="nout")
    circuit.add_capacitor(value="47p", node1="nm", node2="nout")
    circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")
    return circuit

if __name__ == "__main__":
    frequencies = np.logspace(0, 6, 250)

    # Get circuits.
    circuit1 = circuit()
    circuit2 = circuit()
    circuit3 = circuit()

    # Change circuit values.
    circuit2["c1"].capacitance = "1u"
    circuit3["c1"].capacitance = "0.1u"

    # Solve circuits.
    analysis1 = AcSignalAnalysis(circuit=circuit1)
    analysis2 = AcSignalAnalysis(circuit=circuit2)
    analysis3 = AcSignalAnalysis(circuit=circuit3)
    solution1 = analysis1.calculate(frequencies=frequencies, input_type="voltage", node="n1")
    solution2 = analysis2.calculate(frequencies=frequencies, input_type="voltage", node="n1")
    solution3 = analysis3.calculate(frequencies=frequencies, input_type="voltage", node="n1")

    # Give the solutions names (these are used to differentiate the different functions in the final
    # plot).
    solution1.name = "Circuit 1"
    solution2.name = "Circuit 2"
    solution3.name = "Circuit 3"

    # Combine the solutions.
    solution = solution1.combine(solution2, solution3)

    # Plot
    plot = solution.plot_responses(sink="nout", groups="all")
    plot.show()
