"""Circuit simulator command line interface"""

import sys
import logging
from click import Path, File, group, argument, option, version_option, pass_context

from . import __version__, PROGRAM, DESCRIPTION, set_log_verbosity
from .liso import LisoInputParser, LisoOutputParser, LisoRunner, LisoParserError

LOGGER = logging.getLogger(__name__)

# Shared arguments:
# https://github.com/pallets/click/issues/108

class State:
    """CLI state"""
    MIN_VERBOSITY = logging.WARNING
    MAX_VERBOSITY = logging.DEBUG

    def __init__(self):
        self._verbosity = self.MIN_VERBOSITY

    @property
    def verbosity(self):
        """Verbosity on stdout"""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = self.MIN_VERBOSITY - 10 * int(verbosity)

        if self._verbosity < self.MAX_VERBOSITY:
            self._verbosity = self.MAX_VERBOSITY

        set_log_verbosity(self._verbosity)

    @property
    def verbose(self):
        """Verbose output enabled

        Returns True if the verbosity is enough for INFO or DEBUG messages to be displayed.
        """
        return self.verbosity <= logging.INFO

def set_verbosity(ctx, _, value):
    """Set stdout verbosity"""
    state = ctx.ensure_object(State)
    state.verbosity = value

@group(help=DESCRIPTION)
@version_option(version=__version__, prog_name=PROGRAM)
@option("-v", "--verbose", count=True, default=0, callback=set_verbosity, expose_value=False,
        help="Enable verbose output. Supply extra flag for greater verbosity, i.e. \"-vv\"")
def cli():
    """Base CLI command group"""
    pass

@cli.command()
@argument("file", type=File())
@option("--liso", "compute_liso", default=False, help="Simulate using LISO.")
@option("--liso-compare", is_flag=True, default=False,
        help="Simulate using both this tool and LISO binary, and overlay results.")
@option("--liso-diff", is_flag=True, default=False,
        help="Show difference between results of --liso-compare.")
@option("--liso-path", type=Path(exists=True, dir_okay=False), envvar='LISO_PATH',
        help="Path to LISO binary.")
@option("--plot/--no-plot", default=True, show_default=True, help="Display results as figure.")
@option("--save-figure", type=File("wb", lazy=False), help="Save image of figure to file.")
@option("--prescale/--no-prescale", default=True, show_default=True,
        help="Prescale matrices to improve numerical precision.")
@option("--print-equations", is_flag=True, help="Print circuit equations.")
@option("--print-matrix", is_flag=True, help="Print circuit matrix.")
@pass_context
def liso(ctx, file, compute_liso, liso_compare, liso_diff, liso_path, plot, save_figure, prescale,
         print_equations, print_matrix):
    """Parse and simulate LISO input or output file"""
    state = ctx.ensure_object(State)

    # check if native solution must be computed
    compute_native = liso_compare or not compute_liso

    if compute_liso:
        # run file with LISO and parse results
        runner = LisoRunner(script_path=file.name)
        parser = runner.run(liso_path, plot=False, parse_output=compute_native)
        liso_solution = parser.solution()
    else:
        # parse specified file
        try:
            # try to parse as input file
            parser = LisoInputParser()
            parser.parse(path=file.name)
        except LisoParserError:
            try:
                # try to parse as an output file
                parser = LisoOutputParser()
                parser.parse(path=file.name)
            except LisoParserError:
                raise ValueError("cannot interpret specified file as either a LISO input or LISO "
                                 "output file")

    if compute_native:
        # build argument list
        kwargs = {"prescale": prescale,
                  "print_progress": state.verbose,
                  "print_equations": print_equations,
                  "print_matrix": print_matrix}

        # get native solution
        native_solution = parser.solution(force=True, **kwargs)

    # determine solution to show or save
    if liso_compare:
        # combine results from LISO and native simulations
        solution = liso_solution + native_solution

        #with native_solution.plot_style_context({'lines.linestyle': "--"}):

        if liso_diff:
            # TODO: show difference
            pass
    else:
        # plot single result
        if compute_liso:
            # use LISO's solution
            solution = liso_solution
        else:
            # use native solution
            solution = native_solution

    # determine whether to generate plot
    generate_plot = plot or save_figure

    if generate_plot:
        if solution.has_tfs:
            figure = solution.plot_tfs()
        else:
            figure = solution.plot_noise()

        if save_figure:
            # there should only be one figure produced in CLI mode
            solution.save_figure(figure, save_figure)

    if plot:
        solution.show()
