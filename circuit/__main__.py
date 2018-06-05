#!/usr/bin/env python3

"""Circuit simulator utility"""

import io
import sys
import abc
import logging
import argparse
import textwrap
import collections

from circuit import __version__, DESCRIPTION, PROGRAM, logging_on
from .liso.input import LisoInputParser
from .liso.output import LisoOutputParser
from .liso.runner import LisoRunner

LOGGER = logging.getLogger()

parser = argparse.ArgumentParser()

# create subparsers, storing subcommand string
subparsers = parser.add_subparsers(dest="subcommand")

verbose = argparse.ArgumentParser(add_help=False)
verbose.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")

liso_meta = argparse.ArgumentParser(add_help=False)
liso_meta.add_argument("path", help="file path")

liso_in_or_out = argparse.ArgumentParser(add_help=False)
liso_in_or_out.add_argument("--force-input", action="store_true", help="force parsing as LISO input file")
liso_in_or_out.add_argument("--force-output", action="store_true", help="force parsing as LISO output file")

# interpret a LISO file, then run it
liso_native_parser = subparsers.add_parser("liso", help="parse and run a LISO input or output file",
                                           parents=[verbose, liso_meta, liso_in_or_out])

# interpret a LISO file, run it, then compare it to LISO's results
liso_compare_parser = subparsers.add_parser("liso-compare", help="parse and run a LISO input or output file, "
                                            "and show a comparison to LISO's own results",
                                            parents=[verbose, liso_meta, liso_in_or_out])

# run LISO directly
liso_external_parser = subparsers.add_parser("liso-external", help="run an input file with a local LISO binary "
                                             "and show its results",
                                             parents=[verbose, liso_meta])
liso_external_parser.add_argument("--liso-plot", action="store_true", 
                                  help="allow LISO to plot its results")

# LISO path settings
liso_path_parser = subparsers.add_parser("liso-path", help="show or set LISO binary path")
liso_path_parser.add_argument("--set-path", action="store", help="set path to LISO binary")

def action(namespace):
    if hasattr(namespace, "verbose") and namespace.verbose:
        logging_on()

    if namespace.subcommand:
        subcommand = namespace.subcommand

        if subcommand == "liso":
            if namespace.force_output:
                liso_parser = LisoOutputParser()
                liso_parser.parse(namespace.path)
            else:
                try:
                    liso_parser = LisoInputParser()
                    liso_parser.parse(namespace.path)
                except SyntaxError:
                    # file is invalid as input
                    if namespace.force_input:
                        # don't continue
                        raise

                    # try as output
                    liso_parser = LisoOutputParser()
                    liso_parser.parse(namespace.path)
            
            liso_parser.show()
        elif subcommand == "liso-compare":
            # compare native simulation to LISO

            # LISO runner
            runner = LisoRunner(namespace.path)

            # run LISO, and parse its output
            liso_parser = runner.run()

            # get LISO solution
            liso_solution = liso_parser.solution()

            # get native solution
            native_solution = liso_parser.solution(force=True)

            # compare
            figure = liso_solution.bode_figure()
            liso_solution.plot_tfs(figure=figure)
            native_solution.plot_tfs(figure=figure)
            liso_solution.show()
        elif subcommand == "liso-external":
            # run LISO directly and plot its results
            runner = LisoRunner(namespace.path)
            
            # run
            solution = runner.run(namespace.liso_plot)

            if not namespace.liso_plot:
                solution.show()
        elif subcommand == "liso-path":
            if namespace.set_path:
                raise NotImplementedError("this feature is not yet available")
            else:
                runner = LisoRunner()
                print(runner.liso_path)

def main():
    """Main program"""

    # parse arguments
    namespace = parser.parse_args(sys.argv[1:])

    # conduct action
    action(namespace)

if __name__ == "__main__":
    main()
