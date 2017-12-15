"""Parses LISO output file as a solution and plots results."""

import os.path
from electronics.simulate.liso import OutputParser

# check LISO output exist
if not os.path.exists("liso1.out"):
    raise FileNotFoundError("please run `liso1.py` example first")

# parse
parser = OutputParser("liso1")

# solution
solution = parser.solution

# plot
solution.plot()
solution.show()
