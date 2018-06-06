"""Runs local LISO binary on a LISO input file, and plots its output."""

import os
from circuit import logging_on
logging_on()
from circuit.liso import LisoRunner

# run
output = LisoRunner("liso1.fil").run()
solution = output.solution()
solution.plot()
solution.show()
