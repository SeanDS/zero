"""Parses LISO output file specified as an argument to the program, then plots
the results. If no file is specified, "liso1.out" is used."""

import sys
from circuit import logging_on
logging_on()
from circuit.liso import LisoOutputParser

# parse liso filename, if present
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "liso1.out"

# parse output file
parser = LisoOutputParser()
parser.parse(filename)
# show results
parser.show()
