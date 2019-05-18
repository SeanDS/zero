
"""Documentation example. Requires target figure filename as argument."""

import sys
from zero.liso import LisoInputParser

# create parser
parser = LisoInputParser()

base_circuit = """
l l2 420n nlf nout
c c4 47p nlf nout
c c1 1n nrf gnd
r r1 1k nrf gnd
l l1 600n nrf n_l1_c2
c c2 330p n_l1_c2 n_c2_c3
c c3 33p n_c2_c3 nout
c load 20p nout gnd

freq log 100k 100M 1000
uoutput nout
"""

# parse base circuit
parser.parse(base_circuit)
# set input to low frequency port
parser.parse("uinput nlf 50")
# ground unused input
parser.parse("r nrfsrc 5 nrf gnd")
# calculate solution
solutionlf = parser.solution()
solutionlf.name = "LF Circuit"

# reset parser state
parser.reset()

# parse base circuit
parser.parse(base_circuit)
# set input to radio frequency port
parser.parse("uinput nrf 50")
# ground unused input
parser.parse("r nlfsrc 5 nlf gnd")
# calculate solution
solutionrf = parser.solution()
solutionrf.name = "RF Circuit"

# combine solutions
solution = solutionlf.combine(solutionrf)

# plot
plot = solution.plot_responses()
solution.save_figure(plot, sys.argv[1])
