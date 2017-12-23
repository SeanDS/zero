"""Runs local LISO binary on a circuit file."""

import os
from electronics import logging_on
logging_on()
from electronics.simulate.liso import Runner

# run
output = Runner("liso1.fil").run()
output.solution().plot()
