"""Parses LISO output file as a solution and plots results."""

from circuit import logging_on
logging_on()
from circuit.liso import OutputParser

# parse
parser = OutputParser("liso1.out")
parser.show()
