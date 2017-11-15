"""Parses LISO file and simulates the resulting circuit.

Note that LISO syntax is not fully supported, especially the plotting
commands (e.g. uoutput). Instead, the solver provides a `Solution`
object which can be called to plot a transfer function from the input
to any other node."""

import logging
import numpy as np

# enable logging to stdout
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s'))
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from electronics.liso import CircuitParser

# frequency vector
frequencies = np.logspace(0, 6, 1000)

# create parser
parser = CircuitParser()
parser.load("x.fil")

# get circuit from parser
circuit = parser.circuit()
# solve it
solution = parser.circuit().solve(frequencies,
                                  noise_node=circuit.get_node("no"))

# plot
solution.plot_tf()
solution.plot_noise()
solution.show()
