"""Parses LISO output file specified as an argument to the program, then plots
the results. If no file is specified, "liso1.out" is used."""

import sys
import os

from zero.liso import LisoOutputParser

if __name__ == "__main__":
    # parse liso filename, if present
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "liso1.out"

    # convert to file filename
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

    # parse output file
    parser = LisoOutputParser()
    parser.parse(path=filename)
    # simulate and show results
    solution = parser.solution()
    solution.plot()
    solution.show()
