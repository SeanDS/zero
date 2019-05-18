"""Circuit simulator command line interface"""

import sys
import logging
from pprint import pformat
import click
from tabulate import tabulate

from . import __version__, PROGRAM, DESCRIPTION, set_log_verbosity
from .solution import Solution
from .liso import LisoInputParser, LisoOutputParser, LisoRunner, LisoParserError
from .datasheet import PartRequest
from .config import (ZeroConfig, OpAmpLibrary, ConfigDoesntExistException,
                     ConfigAlreadyExistsException, LibraryQueryEngine)

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()
LIBRARY = OpAmpLibrary()


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


def set_verbosity(ctx, _, value):
    """Set stdout verbosity"""
    state = ctx.ensure_object(State)
    state.verbosity = value

@click.group(help=DESCRIPTION)
@click.version_option(version=__version__, prog_name=PROGRAM)
@click.option("-v", "--verbose", count=True, default=0, callback=set_verbosity, expose_value=False,
              help="Enable verbose output. Supply extra flag for greater verbosity, i.e. \"-vv\".")
def cli():
    """Base CLI command group"""
    pass

@cli.command()
@click.argument("file", type=click.File())
@click.option("--liso", is_flag=True, default=False, help="Simulate using LISO.")
@click.option("--liso-path", type=click.Path(exists=True, dir_okay=False), envvar='LISO_PATH',
              help="Path to LISO binary. If not specified, the environment variable LISO_PATH is "
              "searched.")
@click.option("--compare", is_flag=True, default=False,
              help="Simulate using both this tool and LISO binary, and overlay results.")
@click.option("--diff", is_flag=True, default=False,
              help="Show difference between results of comparison.")
@click.option("--plot/--no-plot", default=True, show_default=True, help="Display results as "
              "figure.")
@click.option("--save-figure", type=click.File("wb", lazy=False), multiple=True,
              help="Save image of figure to file. Can be specified multiple times.")
@click.option("--print-equations", is_flag=True, help="Print circuit equations.")
@click.option("--print-matrix", is_flag=True, help="Print circuit matrix.")
@click.pass_context
def liso(ctx, file, liso, liso_path, compare, diff, plot, save_figure, print_equations,
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
        liso_solution.name = "LISO"
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
        kwargs = {"print_progress": state.verbose,
                  "print_equations": print_equations,
                  "print_matrix": print_matrix}

        # get native solution
        native_solution = parser.solution(force=True, **kwargs)
        native_solution.name = "Native"

    # determine solution to show or save
    if compare:
        liso_functions = liso_solution.default_functions[Solution.DEFAULT_GROUP_NAME]
        def liso_order(function):
            """Return order as specified in LISO file for specified function"""
            for index, liso_function in enumerate(liso_functions):
                if liso_function.meta_equivalent(function):
                    return index

            raise ValueError(f"{function} is not in LISO solution")

        # Sort native solution in the order defined in the LISO file.
        native_solution.sort_functions(liso_order, default_only=True)

        # show difference before changing labels
        if diff:
            # group by meta data
            header, rows = native_solution.difference(liso_solution, defaults_only=True,
                                                      meta_only=True)

            click.echo(tabulate(rows, header, tablefmt=CONF["format"]["table"]))

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
        if solution.has_responses:
            figure = solution.plot_responses()
        else:
            figure = solution.plot_noise()

        if save_figure:
            for save_path in save_figure:
                # NOTE: use figure file's name so that Matplotlib can identify the file type
                # appropriately
                solution.save_figure(figure, save_path.name)

    if plot:
        solution.show()

@cli.group()
def library():
    """Component library functions."""
    pass

@library.command("path")
def library_path():
    """Print component library file path.

    Note: this path may not exist.
    """
    click.echo(click.format_filename(LIBRARY.user_config_path))

@library.command("create")
def library_create():
    """Create empty library file in user directory."""
    # create config
    try:
        LIBRARY.create_user_config()
    except ConfigAlreadyExistsException as e:
        click.echo(e, err=True)
    else:
        click.echo(f"Library created at {LIBRARY.user_config_path}")

@library.command("edit")
def library_edit():
    """Open library file in default editor."""
    try:
        LIBRARY.open_user_config()
    except ConfigDoesntExistException:
        click.echo("Configuration file doesn't exist. Try 'zero library create'.", err=True)

@library.command("remove")
def library_remove():
    """Remove user component library file."""
    path = click.format_filename(LIBRARY.user_config_path)
    click.confirm(f"Delete library file at {path}?", abort=True)
    try:
        LIBRARY.remove_user_config()
    except ConfigDoesntExistException as e:
        click.echo(e, err=True)

@library.command("show")
@click.option("--paged", is_flag=True, default=False, help="Print with paging.")
def library_show(paged):
    """Print the library that Zero uses."""
    echo = click.echo_via_pager if paged else click.echo
    echo(pformat(LIBRARY))

@library.command("search")
@click.argument("query")
@click.option("--a0", is_flag=True, default=False, help="Show open loop gain.")
@click.option("--gbw", is_flag=True, default=False, help="Show gain-bandwidth product.")
@click.option("--vnoise", is_flag=True, default=False, help="Show flat voltage noise.")
@click.option("--vcorner", is_flag=True, default=False, help="Show voltage noise corner frequency.")
@click.option("--inoise", is_flag=True, default=False, help="Show flat current noise.")
@click.option("--icorner", is_flag=True, default=False, help="Show current noise corner frequency.")
@click.option("--vmax", is_flag=True, default=False, help="Show maximum output voltage.")
@click.option("--imax", is_flag=True, default=False, help="Show maximum output current.")
@click.option("--sr", is_flag=True, default=False, help="Show slew rate.")
@click.option("--paged", is_flag=True, default=False, help="Print results with paging.")
def library_search(query, a0, gbw, vnoise, vcorner, inoise, icorner, vmax, imax, sr, paged):
    """Search Zero op-amp library.

    Op-amp parameters listed in the library can be searched:

        model (model name), a0 (open loop gain), gbw (gain-bandwidth product),
        delay, vnoise (flat voltage noise), vcorner (voltage noise corner frequency),
        inoise (flat current noise), icorner (current noise corner frequency),
        vmax (maximum output voltage), imax (maximum output current), sr (slew rate)

    The parser supports basic comparison and logic operators:

        == (equal), != (not equal), > (greater than), >= (greater than or equal),
        < (less than), <= (less than or equal), & (logic AND), | (logic OR)

    Clauses can be grouped together with parentheses:

        (vnoise < 10n & inoise < 10p) | (vnoise < 100n & inoise < 1p)

    The query engine supports arbitrary expressions.

    Example: all op-amps with noise less than 10 nV/sqrt(Hz) and corner frequency
    below 10 Hz:

        vnoise < 10n & vcorner < 10
    """
    echo = click.echo_via_pager if paged else click.echo

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
        click.echo("No op-amps found", err=True)
        sys.exit()

    nmodel = len(devices)
    if nmodel == 1:
        opstr = "op-amp"
    else:
        opstr = "op-amps"

    click.echo(f"{nmodel} {opstr} found:")

    header = ["Model"] + params
    rows = []

    for device in devices:
        row = [device.model]
        row.extend([str(getattr(device, param)) for param in params])
        rows.append(row)

    echo(tabulate(rows, header, tablefmt=CONF["format"]["table"]))

@cli.group()
def config():
    """Zero configuration functions."""
    pass

@config.command("path")
def config_path():
    """Print user config file path.

    Note: this path may not exist.
    """
    click.echo(click.format_filename(CONF.user_config_path))

@config.command("create")
def config_create():
    """Create empty config file in user directory."""
    # create config
    try:
        CONF.create_user_config()
    except ConfigAlreadyExistsException as e:
        click.echo(e, err=True)
    else:
        click.echo(f"Config created at {CONF.user_config_path}")

@config.command("edit")
def config_edit():
    """Open user config file in default editor."""
    try:
        CONF.open_user_config()
    except ConfigDoesntExistException:
        click.echo("Configuration file doesn't exist. Try 'zero config create'.", err=True)

@config.command("remove")
def config_remove():
    """Remove user config file."""
    path = click.format_filename(CONF.user_config_path)
    click.confirm(f"Delete config file at {path}?", abort=True)
    try:
        CONF.remove_user_config()
    except ConfigDoesntExistException as e:
        click.echo(e, err=True)

@config.command("show")
@click.option("--paged", is_flag=True, default=False, help="Print with paging.")
def config_show(paged):
    """Print the config that Zero uses."""
    echo = click.echo_via_pager if paged else click.echo
    echo(pformat(CONF))

@cli.command()
@click.argument("term")
@click.option("-f", "--first", is_flag=True, default=False,
              help="Download first match without further prompts.")
@click.option("--partial/--exact", is_flag=True, default=True, help="Allow partial matches.")
@click.option("--display/--download-only", is_flag=True, default=True,
              help="Display the downloaded file.")
@click.option("-p", "--path", type=click.Path(writable=True),
              help="File or directory in which to save the first found datasheet.")
@click.option("-t", "--timeout", type=click.IntRange(0), help="Request timeout in seconds.")
@click.pass_context
def datasheet(ctx, term, first, partial, display, path, timeout):
    """Search, fetch and display datasheets."""
    state = ctx.ensure_object(State)

    # get parts
    parts = PartRequest(term, partial=partial, path=path, timeout=timeout, progress=state.verbose)

    if not parts:
        click.echo("No parts found", err=True)
        sys.exit()

    if first or len(parts) == 1:
        # latest part
        part = parts.latest_part

        # show results directly
        click.echo(part)
    else:
        click.echo("Found multiple parts:")
        for index, part in enumerate(parts, 1):
            click.echo(click.style(f"{index}: {part}", fg="green"))

        # get selection
        part_choice = click.IntRange(1, len(parts))
        part_index = click.prompt("Enter part number", default=1, type=part_choice)

        # get chosen datasheet
        part = parts[part_index - 1]

    # get chosen part
    if part.n_datasheets == 0:
        click.echo(f"No datasheets found for '{part.mpn}'", err=True)
        sys.exit()

    if first or part.n_datasheets == 1:
        # show results directly
        click.echo(part)

        # get datasheet
        ds = part.latest_datasheet
    else:
        click.echo("Found multiple datasheets:")
        for index, ds in enumerate(part.sorted_datasheets, 1):
            click.echo(click.style(f"{index}: {ds}", fg="green"))

        # get selection
        datasheet_choice = click.IntRange(1, part.n_datasheets)
        datasheet_index = click.prompt("Enter part number", default=1, type=datasheet_choice)

        # get datasheet
        ds = part.datasheets[datasheet_index - 1]

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
