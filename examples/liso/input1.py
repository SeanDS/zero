"""Parses LISO file and simulates the resulting circuit.

Note that LISO syntax is not fully supported, especially the plotting
commands (e.g. uoutput). Instead, the solver provides a `Solution`
object which can be called to plot a transfer function from the input
to any other node."""

from electronics import logging_on
logging_on()
from electronics.simulate.liso import InputParser

# create parser
parser = InputParser("liso1.fil")
parser.show()
