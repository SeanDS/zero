"""Circuit simulator command line interface"""

import sys
import logging
from click import Path, File, IntRange, group, argument, option, version_option, pass_context
from tabulate import tabulate

from . import __version__, PROGRAM, DESCRIPTION, set_log_verbosity
from .liso import LisoInputParser, LisoOutputParser, LisoRunner, LisoParserError
from .datasheet import PartRequest
from .config import ZeroConfig
from .library import LibraryQueryEngine

CONF = ZeroConfig()
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

        # write some debug info now that we've set up the logger
        LOGGER.debug("%s %s", PROGRAM, __version__)

    @property
    def verbose(self):
        """Verbose output enabled

        Returns True if the verbosity is enough for INFO or DEBUG messages to be displayed.
        """
        return self.verbosity <= logging.INFO

    def _print(self, msg, stream, exit_, exit_code):
        print(msg, file=stream)

        if exit_:
            sys.exit(exit_code)

    def print_info(self, msg, exit_=False):
        self._print(msg, sys.stdout, exit_, 0)

    def print_error(self, msg, exit_=True):
        self._print(msg, sys.stderr, exit_, 1)


def set_verbosity(ctx, _, value):
    """Set stdout verbosity"""
    state = ctx.ensure_object(State)
    state.verbosity = value

@group(help=DESCRIPTION)
@version_option(version=__version__, prog_name=PROGRAM)
@option("-v", "--verbose", count=True, default=0, callback=set_verbosity, expose_value=False,
        help="Enable verbose output. Supply extra flag for greater verbosity, i.e. \"-vv\".")
def cli():
    """Base CLI command group"""
    pass

@cli.command()
@argument("file", type=File())
@option("--liso", is_flag=True, default=False, help="Simulate using LISO.")
@option("--liso-path", type=Path(exists=True, dir_okay=False), envvar='LISO_PATH',
        help="Path to LISO binary. If not specified, the environment variable LISO_PATH is searched.")
@option("--compare", is_flag=True, default=False,
        help="Simulate using both this tool and LISO binary, and overlay results.")
@option("--diff", is_flag=True, default=False,
        help="Show difference between results of comparison.")
@option("--plot/--no-plot", default=True, show_default=True, help="Display results as figure.")
@option("--save-figure", type=File("wb", lazy=False), multiple=True,
        help="Save image of figure to file. Can be specified multiple times.")
@option("--prescale/--no-prescale", default=True, show_default=True,
        help="Prescale matrices to improve numerical precision.")
@option("--print-equations", is_flag=True, help="Print circuit equations.")
@option("--print-matrix", is_flag=True, help="Print circuit matrix.")
@pass_context
def liso(ctx, file, liso, liso_path, compare, diff, plot, save_figure, prescale, print_equations,
         print_matrix):
    """Parse and simulate LISO input or output file"""
    state = ctx.ensure_object(State)

    # check which solutions must be computed
    compute_liso = liso or compare
    compute_native = not liso or compare

    if compute_liso:
        # run file with LISO and parse results
        runner = LisoRunner(script_path=file.name)
        parser = runner.run(liso_path, plot=False)
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
    if compare:
        # make LISO solution plots dashed
        for function in liso_solution.functions:
            liso_solution.function_plot_styles[function] = {'lines.linestyle': "--"}

        # show difference before changing labels
        if diff:
            # group by meta data
            header, rows = native_solution.difference(liso_solution, defaults_only=True,
                                                      meta_only=True)

            print(tabulate(rows, header, tablefmt=CONF["format"]["table"]))

        # apply suffix to LISO function labels
        for function in liso_solution.functions:
            function.label_suffix = "LISO"

        # combine results from LISO and native simulations
        solution = native_solution + liso_solution
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
            for save_path in save_figure:
                # NOTE: use figure file's name so that Matplotlib can identify the file type
                # appropriately
                solution.save_figure(figure, save_path.name)

    if plot:
        solution.show()

@cli.command()
@argument("query")
@option("--a0", is_flag=True, default=False, help="Show open loop gain.")
@option("--gbw", is_flag=True, default=False, help="Show gain-bandwidth product.")
@option("--vnoise", is_flag=True, default=False, help="Show flat voltage noise.")
@option("--vcorner", is_flag=True, default=False, help="Show voltage noise corner frequency.")
@option("--inoise", is_flag=True, default=False, help="Show flat current noise.")
@option("--icorner", is_flag=True, default=False, help="Show current noise corner frequency.")
@option("--vmax", is_flag=True, default=False, help="Show maximum output voltage.")
@option("--imax", is_flag=True, default=False, help="Show maximum output current.")
@option("--sr", is_flag=True, default=False, help="Show slew rate.")
@pass_context
def opamp(ctx, query, a0, gbw, vnoise, vcorner, inoise, icorner, vmax, imax, sr):
    """Search Zero op-amp library.

    Op-amp parameters listed in the library can be searched:

        model (model name), a0 (open loop gain), gbw (gain-bandwidth product),
        delay, vnoise (flat voltage noise), vcorner (voltage noise corner frequency),
        inoise (flat current noise), icorner (current noise corner frequency),
        vmax (maximum output voltage), imax (maximum output current), sr (slew rate)

    The parser supports basic comparison and logic operators:

        = (equal), != (not equal), > (greater than), >= (greater than or equal),
        < (less than), <= (less than or equal), & (logic AND), | (logic OR)

    Clauses can be grouped together with parentheses:

        (vnoise < 10n & inoise < 10p) | (vnoise < 100n & inoise < 1p)

    The query engine supports arbitrary expressions.

    Example: all op-amps with noise less than 10 nV/sqrt(Hz) and corner frequency
    below 10 Hz:

        vnoise < 10n & vcorner < 10
    """
    engine = LibraryQueryEngine()

    # build parameter list
    params = []
    if a0:
        params.append("a0")
    if gbw:
        params.append("gbw")
    if vnoise:
        params.append("vnoise")
    if vcorner:
        params.append("vcorner")
    if inoise:
        params.append("inoise")
    if icorner:
        params.append("icorner")
    if vmax:
        params.append("vmax")
    if imax:
        params.append("imax")
    if sr:
        params.append("sr")

    # get results
    devices = engine.query(query)

    if not devices:
        print("No op-amps found")
    else:
        nmodel = len(devices)
        if nmodel == 1:
            opstr = "op-amp"
        else:
            opstr = "op-amps"

        print("%i %s found:" % (nmodel, opstr))

        header = ["Model"] + params
        rows = []

        for device in devices:
            row = [device.model]
            row.extend([str(getattr(device, param)) for param in params])
            rows.append(row)

        print(tabulate(rows, header, tablefmt=CONF["format"]["table"]))

@cli.command()
@argument("term")
@option("-f", "--first", is_flag=True, default=False,
        help="Download first match without further prompts.")
@option("--partial/--exact", is_flag=True, default=True, help="Allow partial matches.")
@option("--display/--download-only", is_flag=True, default=True,
        help="Display the downloaded file.")
@option("-p", "--path", type=Path(writable=True),
        help="File or directory in which to save the first found datasheet.")
@option("-t", "--timeout", type=IntRange(0))
@pass_context
def datasheet(ctx, term, first, partial, display, path, timeout):
    """Search, fetch and display datasheets."""
    state = ctx.ensure_object(State)

    # get parts
    parts = PartRequest(term, partial=partial, path=path, timeout=timeout, progress=state.verbose)

    if not parts:
        state.print_error("No parts found")

    if first or len(parts) == 1:
        # latest part
        part = parts.latest_part

        # show results directly
        state.print_info(part)
    else:
        state.print_info("Found multiple parts:")
        for index, part in enumerate(parts, 1):
            state.print_info("%d: %s" % (index, part))

        chosen_part_idx = 0
        while chosen_part_idx <= 0 or chosen_part_idx > len(parts):
            try:
                chosen_part_idx = int(input("Enter part number: "))

                if chosen_part_idx <= 0 or chosen_part_idx > len(parts):
                    raise ValueError
            except ValueError:
                state.print_error("Invalid, try again", exit=False)

        # get chosen datasheet
        part = parts[chosen_part_idx - 1]

    # get chosen part
    if part.n_datasheets == 0:
        state.print_error("No datasheets found for '%s'" % part.mpn)

    if first or part.n_datasheets == 1:
        # show results directly
        state.print_info(part)

        # get datasheet
        ds = part.latest_datasheet
    else:
        state.print_info("Found multiple datasheets:")
        for index, ds in enumerate(part.sorted_datasheets, 1):
            state.print_info("%d: %s" % (index, ds))

        chosen_datasheet_idx = 0
        while chosen_datasheet_idx <= 0 or chosen_datasheet_idx > part.n_datasheets:
            try:
                chosen_datasheet_idx = int(input("Enter datasheet number: "))

                if chosen_datasheet_idx <= 0 or chosen_datasheet_idx > part.n_datasheets:
                    raise ValueError
            except ValueError:
                state.print_error("Invalid, try again", exit=False)

        # get datasheet
        ds = part.datasheets[chosen_datasheet_idx - 1]

    # display details
    if ds.created is not None:
        LOGGER.debug("Created: %s", ds.created)
    if ds.n_pages is not None:
        LOGGER.debug("Pages: %d", ds.n_pages)
    if ds.url is not None:
        LOGGER.debug("URL: %s", ds.url)

    ds.download()
    LOGGER.debug("Saved to: %s", ds.path)

    if display:
        ds.display()
