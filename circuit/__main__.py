#!/usr/bin/env python3

"""Circuit simulator utility"""

import os
import sys
import logging
import argparse

from circuit import __version__, PROGRAM, DESCRIPTION, logging_on
from .liso import LisoInputParser, LisoOutputParser, LisoRunner, LisoParserError
from .display import NodeGraph

LOGGER = logging.getLogger(__name__)

class Parser(object):
    def __init__(self, program, version, subcommands=None, err_stream=sys.stderr):
        # defaults
        self.parser = None
        self._subparsers = None

        if subcommands is not None:
            # build dict
            subcommands = {command.CMD: command for command in subcommands}

        self.program = program
        self._version = version
        self.subcommands = subcommands
        self.err_stream = err_stream

        self._build_parsers()

    def _build_parsers(self):
        self.parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)
        self.parser.add_argument("--version", action="version", version=self.version)

        # create subparser adder
        self._subparsers = self.parser.add_subparsers(dest="subcommand")

        # add subcommands
        for subcommand in self.subcommands.values():
            subcommand.add(self._subparsers)

    def parse(self, args):
        if len(args) == 1:
            self.print_help(exit=True)

        # parse arguments
        namespace = self.parser.parse_args(sys.argv[1:])

        # conduct action
        self.action(namespace)

    def action(self, namespace):
        if namespace.subcommand:
            subcommand = self.subcommands[namespace.subcommand]
            subcommand.action(namespace)

        # nothing to show
        pass

    def print_help(self, exit=False):
        self.parser.print_help(self.err_stream)

        if exit:
            # exit with error code
            sys.exit(1)

    @property
    def version(self):
        return "{prog} {version}".format(prog=self.program, version=self._version)

class SubCommand(object):
    CMD = None

    def add(self, subparser):
        raise NotImplementedError

    def action(self, namespace):
        raise NotImplementedError

    @property
    def base_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-v", "--verbose", action="store_true",
                            help="enable verbose output")
        
        return parser

    @property
    def liso_path_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("path", help="file path")

        return parser
    
    @property
    def liso_io_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--force-input", action="store_true",
                            help="force parsing as LISO input file")
        parser.add_argument("--force-output", action="store_true",
                            help="force parsing as LISO output file")

        return parser
    
    @property
    def liso_analysisdata_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--print-equations", action="store_true",
                            help="print circuit equations")
        parser.add_argument("--print-matrix", action="store_true",
                            help="print circuit matrix")

        return parser


class Liso(SubCommand):
    CMD = "liso"

    def add(self, subparser):
        return subparser.add_parser(self.CMD, help="parse and run a LISO input or output "
                                                   "file",
                                    parents=[self.base_parser, self.liso_path_parser,
                                             self.liso_io_parser,
                                             self.liso_analysisdata_parser])
    
    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()

        kwargs = {"print_progress": namespace.verbose,
                  "print_equations": namespace.print_equations,
                  "print_matrix": namespace.print_matrix}

        if namespace.force_output:
            liso_parser = LisoOutputParser()
            liso_parser.parse(path=namespace.path)
        else:
            try:
                liso_parser = LisoInputParser()
                liso_parser.parse(path=namespace.path)
            except LisoParserError:
                # file is invalid as input
                if namespace.force_input:
                    # don't continue
                    raise

                # try as output
                liso_parser = LisoOutputParser()
                liso_parser.parse(path=namespace.path)
        
        liso_parser.show(**kwargs)


class LisoCompare(SubCommand):
    CMD = "liso-compare"

    def add(self, subparser):
        return subparser.add_parser(self.CMD, help="parse and run a LISO input file, and "
                                                   "show a comparison to LISO's own results",
                                    parents=[self.base_parser, self.liso_path_parser,
                                             self.liso_analysisdata_parser])

    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()

        kwargs = {"print_progress": namespace.verbose,
                  "print_equations": namespace.print_equations,
                  "print_matrix": namespace.print_matrix}

        # LISO runner
        runner = LisoRunner(namespace.path)

        # run LISO, and parse its output
        liso_parser = runner.run()

        # get LISO solution
        liso_solution = liso_parser.solution()

        # get native solution
        native_solution = liso_parser.solution(force=True, **kwargs)

        # compare
        if liso_solution.has_tfs:
            # compare transfer functions
            # get identified sources and sinks
            sources = liso_parser.default_tf_sources()
            sinks = liso_parser.default_tf_sinks()

            # create figure
            figure = liso_solution.bode_figure()

            # compare
            liso_solution.plot_tfs(figure=figure, sources=sources, sinks=sinks)
            # override line style for native solution
            with native_solution.plot_style_context({'lines.linestyle': "--"}):
                native_solution.plot_tfs(figure=figure, sources=sources, sinks=sinks)
        elif liso_solution.has_noise:
            # compare noise
            # get noise sources used in sum
            sources = liso_parser.displayed_noise_sources

            # the native solution does not compute the "sum" column automatically, so we must
            # ask it to if necessary
            if liso_parser._noise_sum_present:
                sums = liso_parser.summed_noise_sources
            else:
                sums = None

            # create figure
            figure = liso_solution.noise_figure()

            # compare (including sums)
            liso_solution.plot_noise(figure=figure, sources=sources)
            # override line style for native solution
            with native_solution.plot_style_context({'lines.linestyle': "--"}):
                native_solution.plot_noise(figure=figure, sources=sources,
                                           compute_sum_sources=sums)
        else:
            raise Exception("no results were computed")

        liso_solution.show()


class LisoExternal(SubCommand):
    CMD = "liso-external"

    def add(self, subparser):
        parser = subparser.add_parser(self.CMD, help="run an input file with a local LISO "
                                                     "binary and show its results",
                                      parents=[self.base_parser, self.liso_path_parser])
        parser.add_argument("--liso-plot", action="store_true", 
                            help="allow LISO to plot its results")

        return parser

    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()
    
        runner = LisoRunner(namespace.path)
        
        # run
        solution = runner.run(namespace.liso_plot)

        if not namespace.liso_plot:
            solution.show()


class LisoPath(SubCommand):
    CMD = "liso-path"

    def add(self, subparser):
        parser = subparser.add_parser(self.CMD, help="show or set LISO binary path",
                                      parents=[self.base_parser])
        parser.add_argument("--set-path", action="store", help="set path to LISO binary")

        return parser
    
    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()

        if namespace.set_path:
            raise NotImplementedError("this feature is not yet available")
        else:
            runner = LisoRunner()
            print(runner.liso_path)


class LisoGraph(SubCommand):
    CMD = "liso-graph"

    def add(self, subparser):
        return subparser.add_parser(self.CMD, help="show node graph for LISO file",
                                    parents=[self.base_parser, self.liso_path_parser,
                                             self.liso_io_parser,
                                             self.liso_analysisdata_parser])
    
    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()

        kwargs = {"print_progress": namespace.verbose,
                  "print_equations": namespace.print_equations,
                  "print_matrix": namespace.print_matrix}

        if namespace.force_output:
            liso_parser = LisoOutputParser()
            liso_parser.parse(path=namespace.path)
        else:
            try:
                liso_parser = LisoInputParser()
                liso_parser.parse(path=namespace.path)
            except LisoParserError:
                # file is invalid as input
                if namespace.force_input:
                    # don't continue
                    raise

                # try as output
                liso_parser = LisoOutputParser()
                liso_parser.parse(path=namespace.path)
        
        solution = liso_parser.solution(**kwargs)
        
        # create node graph
        graph = NodeGraph(solution.circuit)
        
        # open PDF
        graph.view_pdf()


def main():
    """Main program"""
    parser = Parser(PROGRAM, __version__,
                    subcommands=[Liso(), LisoCompare(), LisoExternal(), LisoPath(), LisoGraph()])
    parser.parse(sys.argv)

if __name__ == "__main__":
    main()
