"""Circuit simulator command line interface"""

import sys
import os
import logging
import csv
from pprint import pformat
import click
from tabulate import tabulate

from . import __version__, PROGRAM, DESCRIPTION, set_log_verbosity
from .solution import Solution
from .liso import LisoInputParser, LisoOutputParser, LisoRunner, LisoParserError
from .datasheet import PartRequest
from .components import OpAmp
from .display import OpAmpVoltageNoisePlotter, OpAmpCurrentNoisePlotter, OpAmpGainPlotter
from .config import (ZeroConfig, OpAmpLibrary, ConfigDoesntExistException,
                     ConfigAlreadyExistsException, LibraryQueryEngine, LibraryParserError)

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()
LIBRARY = OpAmpLibrary()

# Library search filter order.
LIBRARY_FILTER_CHOICE = click.Choice(("ASC", "DESC"), case_sensitive=False)

# Data file formats and their delimiters (single characters).
FILE_FORMAT_DELIMITERS = {"csv": ",", "txt": "\t"}

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
@click.argument("files", type=click.File(), nargs=-1, metavar="[FILE]...")
@click.option("--liso", is_flag=True, default=False, help="Simulate using LISO.")
@click.option("--liso-path", type=click.Path(exists=True, dir_okay=False), envvar='LISO_PATH',
              help="Path to LISO binary. If not specified, the environment variable LISO_PATH is "
              "searched.")
@click.option("--resp-scale-db/--resp-scale-abs", default=True, show_default=True,
              help="Scale response y-axes in decibels.")
@click.option("--compare", is_flag=True, default=False,
              help="Simulate using both this tool and LISO binary, and combine the results.")
@click.option("--diff", is_flag=True, default=False,
              help="Show difference between results of comparison.")
@click.option("--plot/--no-plot", default=True, show_default=True, help="Display results as "
              "figure.")
@click.option("--save-figure", type=click.File("wb", lazy=False), multiple=True,
              help="Save image of figure to file. Can be specified multiple times.")
@click.option("--print-equations", is_flag=True, help="Print circuit equations.")
@click.option("--print-matrix", is_flag=True, help="Print circuit matrix.")
@click.pass_context
def liso(ctx, files, liso, liso_path, resp_scale_db, compare, diff, plot, save_figure,
         print_equations, print_matrix):
    """Parse and simulate LISO input or output file(s). Multiple files can be specified as long as
    they have compatible frequency vectors. These are all simulated and combined into one solution.
    """
    state = ctx.ensure_object(State)

    # Check which solutions must be computed.
    compute_liso = liso or compare
    compute_native = not liso or compare

    if not files:
        click.echo("No input files provided. For help, specify --help.")
        sys.exit(0)

    # Determine whether to add script paths to solution names.
    add_path_suffix = len(files) > 1

    solutions = []

    for liso_file in files:
        if compute_liso:
            if add_path_suffix:
                name_suffix = f" {liso_file.name}"
            else:
                name_suffix = ""
            name = f"LISO{name_suffix}"

            # Run file with LISO and parse results.
            runner = LisoRunner(script_path=liso_file.name)
            parser = runner.run(liso_path, plot=False)
            liso_solution = parser.solution()
            liso_solution.name = name
        else:
            # Parse specified file.
            try:
                # Try to parse as input file.
                parser = LisoInputParser()
                parser.parse(path=liso_file.name)
            except LisoParserError:
                try:
                    # Try to parse as an output file.
                    parser = LisoOutputParser()
                    parser.parse(path=liso_file.name)
                except LisoParserError:
                    click.echo(f"cannot interpret {liso_file.name} as either a LISO input or LISO "
                               "output file", err=True)
                    sys.exit(1)

        if compute_native:
            if add_path_suffix:
                name_suffix = f" {liso_file.name}"
            else:
                name_suffix = ""
            name = f"Zero{name_suffix}"

            # Build argument list.
            kwargs = {"print_progress": state.verbose,
                      "print_equations": print_equations,
                      "print_matrix": print_matrix}

            # Get native solution.
            native_solution = parser.solution(force=True, **kwargs)
            native_solution.name = name

        # Determine solution to show or save.
        if compare:
            liso_functions = liso_solution.default_functions[Solution.DEFAULT_GROUP_NAME]
            def liso_order(function):
                """Return order as specified in LISO file for specified function"""
                for index, liso_function in enumerate(liso_functions):
                    if liso_function.meta_equivalent(function):
                        return index

                click.echo(f"{function} is not in LISO solution", err=True)
                sys.exit(1)

            # Sort native solution in the order defined in the LISO file.
            native_solution.sort_functions(liso_order, default_only=True)

            # Show difference before changing labels.
            if diff:
                # Group by meta data.
                header, rows = native_solution.difference(liso_solution, defaults_only=True,
                                                          meta_only=True)

                click.echo(tabulate(rows, header, tablefmt=CONF["format"]["table"]))

            # Combine results from LISO and native simulations. This puts the functions from each
            # solution into groups with that solution's name so we can differentiate them on the
            # plot.
            solution = native_solution.combine(liso_solution)
        else:
            # Plot single result.
            if compute_liso:
                # Use LISO's solution.
                solution = liso_solution
            else:
                # Use native solution.
                solution = native_solution

        solutions.append(solution)

    solution = solutions[0]
    if len(solutions) > 1:
        # Combine all simulated solutions.
        solution = solution.combine(*solutions[1:], merge_groups=True)

    # Determine whether to generate plot.
    generate_plot = plot or save_figure

    if generate_plot:
        if solution.has_responses:
            plotter = solution.plot_responses(scale_db=resp_scale_db)
        else:
            plotter = solution.plot_noise()

        if save_figure:
            for save_path in save_figure:
                # NOTE: use figure file's name so that Matplotlib can identify the file type
                # appropriately.
                plotter.save(save_path.name)

    if plot:
        plotter.show()

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
@click.option("--sort-a0", type=LIBRARY_FILTER_CHOICE, default="DESC", show_default=True)
@click.option("--sort-gbw", type=LIBRARY_FILTER_CHOICE, default="DESC", show_default=True)
@click.option("--sort-delay", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--sort-vnoise", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--sort-vcorner", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--sort-inoise", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--sort-icorner", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--sort-vmax", type=LIBRARY_FILTER_CHOICE, default="DESC", show_default=True)
@click.option("--sort-imax", type=LIBRARY_FILTER_CHOICE, default="DESC", show_default=True)
@click.option("--sort-sr", type=LIBRARY_FILTER_CHOICE, default="ASC", show_default=True)
@click.option("--show-table/--no-show-table", default=True, show_default=True,
              help="Print results as a table.")
@click.option("--paged", is_flag=True, default=False, help="Print results with paging.")
@click.option("--save-data", type=click.File("wb", lazy=False), multiple=True,
              help="Save search results to file. The file format is decided based on the specified "
                   "extension. Supported extensions are \"csv\" and \"txt\". Can be specified "
                   "multiple times.")
@click.option("--plot-voltage-noise/--no-plot-voltage-noise", is_flag=True, default=False,
              show_default=True, help="Display op-amp voltage noise as figure.")
@click.option("--plot-current-noise/--no-plot-current-noise", is_flag=True, default=False,
              show_default=True, help="Display op-amp current noise as figure.")
@click.option("--plot-gain/--no-plot-gain", is_flag=True, default=False,
              show_default=True, help="Display op-amp open loop gain as figure.")
@click.option("--save-voltage-noise-figure", type=click.File("wb", lazy=False), multiple=True,
              help="Save image of voltage noise figure to file. Can be specified multiple times.")
@click.option("--save-current-noise-figure", type=click.File("wb", lazy=False), multiple=True,
              help="Save image of current noise figure to file. Can be specified multiple times.")
@click.option("--save-gain-figure", type=click.File("wb", lazy=False), multiple=True,
              help="Save image of open loop gain figure to file. Can be specified multiple times.")
@click.option("--fstart", type=str, default="1", show_default=True, help="Plot start frequency.")
@click.option("--fstop", type=str, default="1G", show_default=True, help="Plot stop frequency.")
@click.option("--npoints", type=int, default=1000, show_default=True, help="Plot number of points.")
def library_search(query, sort_a0, sort_gbw, sort_delay, sort_vnoise, sort_vcorner, sort_inoise,
                   sort_icorner, sort_vmax, sort_imax, sort_sr, show_table, paged, save_data,
                   plot_voltage_noise, plot_current_noise, plot_gain, save_voltage_noise_figure,
                   save_current_noise_figure, save_gain_figure, fstart, fstop, npoints):
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

    The 'a0' parameter can be specified in magnitude or decibels. For decibels, append 'dB' (case
    insensitive) to the value.

    The results are sorted sequentially in the order that each parameter appears in the
    search query (left to right). The sort direction (descending or ascending) depends on the type
    of parameter. The sort direction for parameter 'X' can be overridden using the corresponding
    '--sort-X' flag. Specify 'ASC' for ascending and 'DESC' for descending order.
    """
    engine = LibraryQueryEngine()
    # Define sort order based on defaults and user preferences. Models are always alphabetical.
    sort_order = {"model": False, "a0": sort_a0 == "DESC", "gbw": sort_gbw == "DESC",
                  "delay": sort_delay == "DESC", "vnoise": sort_vnoise == "DESC",
                  "vcorner": sort_vcorner == "DESC", "inoise": sort_inoise == "DESC",
                  "icorner": sort_icorner == "DESC", "vmax": sort_vmax == "DESC",
                  "imax": sort_imax == "DESC", "sr": sort_sr == "DESC"}
    # Get results.
    try:
        devices = engine.query(query, sort_order=sort_order)
    except (LibraryParserError, ValueError) as error:
        click.echo(str(error), err=True)
        click.echo("Add --help for syntax help.", err=True)
        sys.exit(1)

    if not devices:
        click.echo("No op-amps found.")
        sys.exit(0)

    nmodel = len(devices)

    if nmodel == 1:
        opstr = "op-amp"
    else:
        opstr = "op-amps"

    opamps = []
    rows = []

    for device in devices:
        rows.append([str(getattr(device, param)) for param in engine.parameters])

        opamp = OpAmp(model=OpAmpLibrary.format_name(device.model), node1="input", node2="gnd",
                      node3="output", **LIBRARY.get_data(device.model))
        opamps.append(opamp)

    table = tabulate(rows, engine.parameters, tablefmt=CONF["format"]["table"])
    if show_table:
        click.echo(f"{nmodel} {opstr} found:")
        if paged:
            click.echo_via_pager(table)
        else:
            click.echo(table)

    if save_data:
        for path in save_data:
            pieces = os.path.splitext(path.name)
            if not len(pieces) == 2:
                click.echo(f"Path {path} extension invalid.", err=True)
                sys.exit(1)
            # Remove leading full stop.
            extension = pieces[1][1:]
            if extension.lower() not in FILE_FORMAT_DELIMITERS:
                click.echo(f"File format '{extension}' not recognised.", err=True)
                sys.exit(1)
            delimiter = FILE_FORMAT_DELIMITERS[extension]
            with open(path.name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow(engine.parameters)
                writer.writerows(rows)

    def _plot_save_figure(plot_flag, save_flag, plot_type):
        """Plot and/or save an op-amp plot of a particular type."""
        if plot_flag or save_flag:
            plotter = plot_type(fstart=fstart, fstop=fstop, npoints=npoints)
            plotter.plot(opamps)

            if save_flag:
                for save_path in save_flag:
                    # NOTE: use figure file's name so that Matplotlib can identify the file type
                    # appropriately.
                    plotter.save(save_path.name)

            if plot_flag:
                plotter.show()

    _plot_save_figure(plot_voltage_noise, save_voltage_noise_figure, OpAmpVoltageNoisePlotter)
    _plot_save_figure(plot_current_noise, save_current_noise_figure, OpAmpCurrentNoisePlotter)
    _plot_save_figure(plot_gain, save_gain_figure, OpAmpGainPlotter)


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
