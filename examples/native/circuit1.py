"""Native circuit construction and simulation."""

import numpy as np
from electronics import logging_on
logging_on()
from electronics.simulate.circuit import Circuit
from electronics.simulate.components import Resistor, Capacitor, OpAmp, Node

# frequency vector
frequencies = np.logspace(0, 6, 1000)

# create circuit object
circuit = Circuit()

# add components
circuit.add_capacitor(name="c1", value=10e-6, node1="gnd", node2="n1")
circuit.add_resistor(name="r1", value=430, node1="n1", node2="nm")
circuit.add_resistor(name="r2", value=43e3, node1="nm", node2="nout")
circuit.add_capacitor(name="c2", value=47e-12, node1="nm", node2="nout")
circuit.add_library_opamp(name="o1", model="LT1124", node1="nin", node2="nm",
                          node3="nout")

# solve circuit
solution = circuit.solve(frequencies, input_node_p="nin", input_impedance=0,
                         output_components="all", output_nodes="all",
                         noise_node="nout")

print("Circuit matrix for f = %d" % frequencies[0])
circuit.print_matrix(frequency=frequencies[0])
#print("Circuit equations for f = %d" % frequencies[0])
#circuit.print_equations(frequency=frequencies[0])

# plot
solution.plot()
