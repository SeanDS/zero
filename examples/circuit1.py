"""Native circuit construction and simulation."""

import logging
import numpy as np

# enable logging to stdout
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from electronics.simulate.circuit import Circuit
from electronics.simulate.components import (Resistor, Capacitor, OpAmp, Node,
                                             Gnd)
from electronics.config import OpAmpLibrary as lib

# frequency vector
frequencies = np.logspace(0, 6, 1000)

# create nodes
gnd = Gnd()
n1 = Node("n1")
nm = Node("nm")
nout = Node("nout")
nin = Node("nin")

c1 = Capacitor(name="c1", value=10e-6, node1=gnd, node2=n1)
r1 = Resistor(name="r1", value=430, node1=n1, node2=nm)
r2 = Resistor(name="r2", value=43e3, node1=nm, node2=nout)
c2 = Capacitor(name="c2", value=47e-12, node1=nm, node2=nout)
op = OpAmp(name="o1", model="LT1124", node1=nin, node2=nm, node3=nout)

# create circuit with input node
circuit = Circuit()
# add components
circuit.add_component(c1)
circuit.add_component(r1)
circuit.add_component(r2)
circuit.add_component(c2)
circuit.add_component(op)

# solve circuit
solution = circuit.solve(frequencies, input_nodes=[nin], noise_node=nout)

print("Circuit matrix for f = %d" % frequencies[0])
circuit.print_matrix(frequency=frequencies[0])
print("Circuit equations for f = %d" % frequencies[0])
circuit.print_equations(frequency=frequencies[0])

# plot
solution.plot_tf()
solution.plot_noise()
solution.show()
