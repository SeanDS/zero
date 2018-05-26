"""Runs local LISO binary on a circuit file."""

import os
from circuit import logging_on
logging_on()
from circuit.liso.runner import Runner

# run
output = Runner("liso1.fil").run()
output.solution().plot()
