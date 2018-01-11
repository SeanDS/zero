"""Parses LISO input file specified as an argument to the program, then
simulates the resulting circuit. If no file is specified, "liso1.fil" is
used."""

import sys
from circuit import logging_on
logging_on()
from circuit.liso import InputParser

# parse liso filename, if present
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "liso1.fil"

# parse input file
parser = InputParser(filename)
# simulate and show results
parser.show(print_equations=True, print_matrix=True)
