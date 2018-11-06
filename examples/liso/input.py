"""Parses LISO input file specified as an argument to the program, then
simulates the resulting circuit. If no file is specified, "liso1.fil" is
used."""

import sys
import os

from zero.liso import LisoInputParser

if __name__ == "__main__":
    # parse liso filename, if present
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "liso1.fil"

    # convert to file filename
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    # parse input file
    parser = LisoInputParser()
    parser.parse(path=filename)
    # simulate and show results
    solution = parser.solution()
    solution.plot()
    solution.show()
