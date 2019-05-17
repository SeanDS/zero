"""Documentation example. Requires target figure filename as argument."""

import sys
from zero.liso import LisoInputParser

parser = LisoInputParser()

parser.parse("""
c c1 10u gnd n1
r r1 430 n1 nm
r r2 43k nm nout
c c2 47p nm nout
op o1 lt1124 nin nm nout

freq log 1 100k 100

uinput nin 0
uoutput nout:db:deg
""")

solution = parser.solution()
plot = solution.plot_responses()
solution.save_figure(plot, sys.argv[1])
