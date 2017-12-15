"""Parses LISO output file as a solution and plots results."""

import os.path
from electronics import logging_on
logging_on()
from electronics.simulate.liso import OutputParser, Runner

# parse
parser = OutputParser("liso1.fil")

# solution
solution = parser.solution

# plot
solution.plot()
solution.show()
