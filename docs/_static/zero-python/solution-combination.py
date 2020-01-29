
"""Documentation example. Requires target figure filename as argument."""

import sys
from zero.liso import LisoInputParser

# Create parser.
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

# Parse the base circuit.
parser.parse(base_circuit)
# Set the circuit input to the low frequency port.
parser.parse("uinput nlf 50")
# Ground the unused input.
parser.parse("r nrfsrc 5 nrf gnd")
# Calculate the solution.
solutionlf = parser.solution()
solutionlf.name = "LF Circuit"

# Reset the parser's state.
parser.reset()

# Parse the base circuit.
parser.parse(base_circuit)
# Set the input to the radio frequency port.
parser.parse("uinput nrf 50")
# Ground the unused input.
parser.parse("r nlfsrc 5 nlf gnd")
# Calculate the solution.
solutionrf = parser.solution()
solutionrf.name = "RF Circuit"

# Combine the solutions. By default, this keeps the functions from each source solution in different
# groups in the resulting solution. This makes the plot show the functions with different styles and
# shows the source solution's name as a suffix on each legend label.
solution = solutionlf.combine(solutionrf)

# Plot.
plotter = solution.plot_responses()
plotter.save(sys.argv[1])
