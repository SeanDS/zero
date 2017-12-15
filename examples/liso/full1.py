"""Runs local LISO binary on a circuit file."""

import os
from electronics import logging_on
logging_on()
from electronics.simulate.liso import Runner

# check LISO output exist
if not os.path.exists("liso1.out"):
    # run LISO to produce output file
    Runner("liso1.fil").run(output_path="liso1.out")

# run
output = Runner("liso1.fil").run()
output.solution.plot()
output.solution.show()
