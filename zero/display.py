"""Rich display system"""

import abc
from importlib import import_module
import logging
import collections
import tempfile
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cycler
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate

from .config import ZeroConfig
from .components import Resistor, Capacitor, Inductor, OpAmp, Input, Component, Node
from .format import Quantity
from .data import Series, Response, NoiseDensity, MultiNoiseDensity

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()


def lighten_colours(colour_cycle, factor):
    """Lightens the given color by multiplying (1 - luminosity) by the given factor.

    https://stackoverflow.com/a/49601444/2251982
    """
    cycle = []

    for this_colour in colour_cycle:
        try:
            # get RGB values from hex string or name
            c = colors.cnames[this_colour]
        except KeyError:
            c = this_colour

        c = colorsys.rgb_to_hls(*colors.to_rgb(c))
        new = colorsys.hls_to_rgb(c[0], 1 - factor * (1 - c[1]), c[2])
        newints = tuple([int(value * 255) for value in new])
        hexcode = "#%02x%02x%02x" % newints

        cycle.append(hexcode)

    return cycle


class NodeGraph:
    # input shapes per input type
    input_shapes = {"noise": "ellipse", "voltage": "box", "current": "pentagon"}

    def __init__(self, circuit):
        try:
            self.graphviz = import_module("graphviz")
        except ImportError:
            raise NotImplementedError("Node graph representation requires the graphviz package")

        self.circuit = circuit

    def node_graph(self):
        """Create Graphviz node graph"""

        graph = self.graphviz.Digraph(engine=CONF["graphviz"]["engine"])
        graph.attr("node", style=CONF["graphviz"]["node_style"],
                   fontname=CONF["graphviz"]["node_font_name"],
                   # For some reason, font size must be suppied as string.
                   fontsize=str(CONF["graphviz"]["node_font_size"]))
        graph.attr("edge", arrowhead=CONF["graphviz"]["edge_arrowhead"])
        graph.attr("graph", splines=CONF["graphviz"]["graph_splines"],
                   label="Made with graphviz and Zero",
                   fontname=CONF["graphviz"]["graph_font_name"],
                   # For some reason, font size must be suppied as string.
                   fontsize=str(CONF["graphviz"]["graph_font_size"]))
        node_map = {}

        def add_connection(component, conn, node):
            if node == 'gnd':
                graph.node("%s_%s" % (component, node), shape='point', style='invis')
                graph.edge('%s_%s' % (component, node), component+conn, dir='both', arrowtail='tee',
                           len='0.0', weight='10')
            else:
                if not node in node_map:
                    graph.node(node, shape='point', xlabel=node, width='0.1', fillcolor='Red')

                node_map[node] = node
                graph.edge(node_map[node], component+conn)

        for component in self.circuit.components:
            connections = ['', '']

            if isinstance(component, OpAmp):
                attr = {'shape': 'plain', 'margin': '0', 'orientation': '270'}
                attr['label'] = f"""<<TABLE BORDER="0" BGCOLOR="LightSkyBlue">
                    <TR>
                        <TD PORT="plus">+</TD>
                        <TD ROWSPAN="3">{component.name}<BR/>{component.model}</TD>
                    </TR>
                    <TR>
                        <TD> </TD>
                    </TR>
                    <TR>
                        <TD PORT="minus">-</TD>
                    </TR>
                </TABLE>>"""
                connections = [':plus', ':minus', ':e']
            elif isinstance(component, Inductor):
                attr = {'fillcolor': 'MediumSlateBlue', 'shape': 'diamond'}
            elif isinstance(component, Capacitor):
                attr = {'fillcolor': 'YellowGreen', 'shape': 'diamond'}
            elif isinstance(component, Resistor):
                attr = {'fillcolor': 'Orchid', 'shape': 'diamond'}
            elif isinstance(component, Input):
                attr = {'fillcolor': 'Orange',
                        'shape': self.input_shapes[component.input_type]}
            else:
                raise ValueError("Unrecognised element %s: %s" % (component.name, component.__class__))

            graph.node(component.name, **attr)

            for node, connection in zip(component.nodes, connections):
                add_connection(component.name, connection, node.name)

        return graph

    def view_pdf(self):
        """View the graph as a PDF"""
        return self.node_graph().view(directory=tempfile.gettempdir(), cleanup=True)

    def _repr_svg_(self):
        """Graphviz rendering for Jupyter notebooks."""
        return self.node_graph()._repr_svg_()


class TableFormatter(metaclass=abc.ABCMeta):
    """Table formatter mixin

    Children inheriting this class must implement the `row_cell_groups` method.
    """

    def sanitise_cell(self, cell):
        if isinstance(cell, (str, np.str)):
            # leave strings alone
            return str(cell)

        if isinstance(cell, np.ndarray):
            if len(cell) == 1:
                # this is a single-valued array
                cell = cell[0]
            else:
                raise ValueError("got a multi-valued array when expecting a single number")

        if hasattr(cell, 'real'):
            if np.abs(cell) == 1:
                # get rid of imaginary part, retaining sign
                cell = np.sign(cell.real) * np.abs(cell).real
            else:
                # get rid of imaginary part
                cell = np.abs(cell).real

        # convert to float
        return float(cell)

    @abc.abstractproperty
    def row_cell_groups(self):
        raise NotImplementedError

    @property
    def row_cells(self):
        """Returns an iterable of cells for each row in the table"""
        return [self.combine_cells(*cell_groups) for cell_groups in self.row_cell_groups]

    def combine_cells(self, *cells):
        """Combines the specified collections of cells into a single iterable"""

        for collection in cells:
            if not isinstance(collection, collections.Iterable):
                # convert to list
                collection = [collection]

            yield from collection

    @property
    def table(self):
        """Get unformatted table"""
        return self.row_cells

    @property
    def formatted_table(self):
        """Get formatted table"""
        return [self.format_row(row_cells) for row_cells in self.row_cells]

    def format_row(self, cells):
        yield from [self.format_cell(self.sanitise_cell(cell)) for cell in cells]

    def format_cell(self, cell):
        """Default cell formatter"""
        return cell

    def get_base_power(self, number):
        """Number's base and power"""

        if number == 0:
            raise ValueError("cannot calculate power of 0")

        # number's nearest power of ten
        power = round(np.log10(number))

        # divide down by power
        base = number / 10 ** power

        return base, power


class MatrixDisplay(TableFormatter):
    def __init__(self, lhs, middle, rhs, headers):
        """Instantiate matrix display with extra left and right sides"""
        lhs = np.array(lhs)
        middle = np.array(middle)
        rhs = np.array(rhs)
        headers = list(headers)

        if lhs.shape[1] + middle.shape[1] + rhs.shape[1] != len(headers):
            raise ValueError("lhs + middle + rhs second dimensions must match number of headers")

        self.lhs = lhs
        self.middle = middle
        self.rhs = rhs
        self.headers = headers

        super().__init__()

    @property
    def row_cell_groups(self):
        return zip(self.lhs, self.middle, self.rhs)

    def format_cell(self, cell):
        """Override parent"""

        if cell == 0:
            return "---"

        return cell

    def format_cell_text(self, cell):
        if not isinstance(cell, str):
            if np.abs(cell) == 1:
                # don't format zero or one
                cell = str(int(cell))
            else:
                base, power = self.get_base_power(cell)

                exponent = "e%i" % power

                if power < -12:
                    # just show power for tiny stuff
                    cell = exponent
                else:
                    cell = "%.2f" % base

                    if power != 0:
                        # print non-zero power
                        cell += exponent

        return cell

    def format_cell_html(self, cell):
        if not isinstance(cell, str):
            if np.abs(cell) == 1:
                # don't format zero or one
                cell = str(int(cell))
            else:
                base, power = self.get_base_power(cell)

                if power < -12:
                    # just show power for tiny stuff
                    cell = "10<sup>%i</sup>" % round(np.log10(power))
                else:
                    cell = "%.2f" % base

                    if power != 0:
                        # print non-zero power
                        cell += "×10<sup>%i</sup>" % power

        return "<td>%s</td>" % cell

    def __repr__(self):
        """Text representation of the table"""

        # format table
        table = [[self.format_cell_text(cell) for cell in row] for row in self.formatted_table]

        # tabulate data
        return tabulate(table, self.headers, tablefmt=CONF["format"]["table"])

    def _repr_html_(self):
        """HTML table representation"""

        table = "<table>"
        table += "<thead>"
        table += "<tr>"
        table += "".join(["<th>%s</th>" % header_cell for header_cell in self.headers])
        table += "</tr>"
        table += "</thead>"
        table += "<tbody>"
        table += "".join(["<tr>%s</tr>" % "".join([self.format_cell_html(cell) for cell in row]) for row in self.formatted_table])
        table += "</tbody>"
        table += "</table>"

        return table


class EquationDisplay(TableFormatter):
    def __init__(self, lhs, rhs, elements):
        lhs = np.array(lhs)
        rhs = np.array(rhs)
        elements = list(elements)

        if lhs.shape[1] != len(elements):
            raise ValueError("lhs second dimensions must match number of elements")

        self.lhs = lhs
        self.rhs = rhs
        self.elements = elements

        super().__init__()

    @property
    def row_cell_groups(self):
        return zip(self.lhs, self.rhs)

    def format_coefficient_text(self, coefficient):
        """Format equation coefficient"""

        if coefficient == 0 or abs(coefficient) == 1:
            # don't write zeros or ones
            return ""

        return str(self.format_exponent_text(coefficient))

    def format_coefficient_latex(self, coefficient):
        """Format equation coefficient"""

        if coefficient == 0 or abs(coefficient) == 1:
            # don't write zeros or ones
            return ""

        return str(self.format_exponent_latex(coefficient))

    def format_rhs(self, rhs):
        """Format right hand side number"""

        rhs = self.format_cell(self.sanitise_cell(rhs))

        if rhs == 0:
            # we want to show the right hand side for equations
            return "0"
        elif abs(rhs) == 1:
            # maybe write with sign
            return "%i" % rhs

        return rhs

    def format_exponent_text(self, number):
        """Format number in text"""

        if isinstance(number, str):
            return number

        base, power = self.get_base_power(number)

        if power == 0:
            power_str = ""
        else:
            power_str = "10 ^ %d" % power

        if power < -12:
            # don't print base
            base_str = ""
        else:
            base_str = "%.2f" % base

            if power_str:
                # add multiplication symbol
                base_str += " × "

        return base_str + power_str

    def format_exponent_latex(self, number):
        """Format number in LaTeX"""

        if isinstance(number, str):
            return number

        base, power = self.get_base_power(number)

        if power == 0:
            power_str = ""
        else:
            power_str = r"10^{%d}" % power

        if power < -12:
            # don't print base
            base_str = ""
        else:
            base_str = r"%.2f" % base

            if power_str:
                # add multiplication symbol
                base_str += r"\times"

        return base_str + power_str

    def align_to(self, search, sequence):
        """Line up elements in `sequence` to `search` character or string"""

        # find longest line up to equals
        max_pos = max([item.find(search) for item in sequence])

        # prepend whitespace to each item
        for item in sequence:
            # number of spaces to add
            n_spaces = max_pos - item.find(search)

            yield " " * n_spaces + item

    def __repr__(self):
        """Text representation of equations"""

        lines = []

        for lhs_coefficients, rhs_value in zip(self.formatted_table, self.rhs):
            # flag to suppress leading sign
            first = True

            clauses = []

            # loop over equation coefficients
            for coefficient, element in zip(lhs_coefficients, self.elements):
                clause = ""

                if coefficient == 0:
                    # don't print
                    continue

                if np.sign(coefficient) == -1:
                    # add negative sign
                    clause += "- "
                elif not first:
                    # add positive sign
                    clause += "+ "

                # flag that we're beyond the first column
                first = False

                # format element
                if isinstance(element, Component):
                    # current through component
                    element_format = "I[%s]"
                elif isinstance(element, Node):
                    element_format = "V[%s]"
                else:
                    raise ValueError("unexpected element type")

                # format coefficient
                formatted_coefficient = self.format_coefficient_text(coefficient)

                if formatted_coefficient:
                    formatted_coefficient += " × "

                # write coefficient and element
                clause += "%s%s" % (formatted_coefficient,  element_format % element.label)

                clauses.append(clause)

            # add right hand side, with alignment character
            clauses.append("= %s" % self.format_exponent_text(self.format_rhs(rhs_value)))

            # make line from clauses
            lines.append(" ".join(clauses))

        lines = self.align_to("=", lines)

        return "\n".join(lines)

    def _repr_latex_(self):
        """LaTeX representation of equations"""

        expression = r"\begin{align}"

        for lhs_coefficients, rhs_value in zip(self.formatted_table, self.rhs):
            # flag to suppress leading sign
            first = True

            # loop over equation coefficients
            for coefficient, element in zip(lhs_coefficients, self.elements):
                if coefficient == 0:
                    # don't print
                    continue

                if np.sign(coefficient) == -1:
                    # add negative sign
                    expression += r"-"
                elif not first:
                    # add positive sign
                    expression += r"+"

                # flag that we're beyond the first column
                first = False

                # format element
                if isinstance(element, Component):
                    # current through component
                    element_format = r"I_{%s}"
                elif isinstance(element, Node):
                    element_format = r"V_{%s}"
                else:
                    raise ValueError("unexpected element type")

                # write coefficient and element
                expression += self.format_coefficient_latex(coefficient) + element_format % element.label

            # add right hand side, with alignment character
            expression += r"&="
            expression += self.format_exponent_latex(self.format_rhs(rhs_value))

            # add newline
            expression += r"\\"

        expression += r"\end{align}"

        return expression


class BasePlotter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def plot(self, functions, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def show(self):
        raise NotImplementedError


class BaseGroupPlotter(metaclass=abc.ABCMeta):
    def __init__(self, legend_groups=True, hidden_group_names=None, **kwargs):
        super().__init__(**kwargs)
        self.legend_groups = legend_groups
        if hidden_group_names is None:
            hidden_group_names = []
        self.hidden_group_names = list(hidden_group_names)

    @abc.abstractmethod
    def plot_groups(self, groups):
        raise NotImplementedError


class MatplotlibPlotter(BasePlotter, metaclass=abc.ABCMeta):
    def __init__(self, figure=None, title=None, legend=True, legend_loc="best", **kwargs):
        super().__init__(**kwargs)
        # Defaults.
        self._figure = None
        # Parameters.
        if figure is not None:
            self.figure = figure
        self.title = title
        self.legend = legend
        self.legend_loc = legend_loc

    def _expand_linear_axis_limits(self, dmin, dmax, step, margin):
        """Intelligently expand specified limits for a linearly scaled axis.

        This is similar in behaviour to Matplotlib when axes.autolimit_mode = round_numbers, except
        it applies the autoscale margin rcparam to the current y-axis limits before deciding whether
        to add another major tick step to the limits.

        The reason why this method might be used instead of the default Matplotlib behaviour is to
        ensure that when axis data is close to the upper or lower limit of its axis, the axis limits
        are expanded by a full `step` on top of the closest higher/lower step (for upper/lower
        limits, respectively). By default, Matplotlib will simply expand by the margin settings,
        which is by default 5%. Expanding by a full step is in some contexts neater.
        """
        # closeto, le and ge adapted from :class:`matplotlib.ticker._Edge_integer` (3.2.0).
        def closeto(ms, edge):
            tol = 1e-10
            return abs(ms - edge) < tol

        def le(x):
            d, m = divmod(x, step)
            if closeto(m / step, 1):
                return (d + 1)
            return d

        def ge(x):
            d, m = divmod(x, step)
            if closeto(m / step, 0):
                return d
            return (d + 1)

        factor = 1 + margin

        vmin = le(dmin * factor) * step
        vmax = ge(dmax * factor) * step
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        return vmin, vmax

    def _expand_log_axis_limits(self, dmin, dmax, base, margin):
        """Intelligently expand specified limits for a log scaled axis.

        This is similar in behaviour to Matplotlib when axes.autolimit_mode = round_numbers, except
        it applies the autoscale margin rcparam to the current y-axis limits before deciding whether
        to add another major tick step to the limits.

        The reason why this method might be used instead of the default Matplotlib behaviour is to
        ensure that when axis data is close to the upper or lower limit of its axis, the axis limits
        are expanded by a full `base` on top of the closest higher/lower multiple of `base` (for
        upper/lower limits, respectively). By default, Matplotlib will simply expand by the margin
        settings, which is by default 5%. Expanding by a full step is in some contexts neater.
        """
        def _decade_less_equal(x, base):
            return (x if x == 0 else
                    -_decade_greater_equal(-x, base) if x < 0 else
                    base ** np.floor(np.log(x) / np.log(base)))

        def _decade_greater_equal(x, base):
            return (x if x == 0 else
                    -_decade_less_equal(-x, base) if x < 0 else
                    base ** np.ceil(np.log(x) / np.log(base)))

        factor = 1 + margin

        vmin = _decade_less_equal(dmin * factor, base)
        vmax = _decade_greater_equal(dmax * factor, base)

        return vmin, vmax

    @property
    def figure(self):
        if self._figure is None:
            self._figure = self._create_figure()
        return self._figure

    @figure.setter
    def figure(self, figure):
        self._figure = figure

    def _create_figure(self):
        figure = plt.figure(figsize=(float(CONF["plot"]["size_x"]), float(CONF["plot"]["size_y"])))
        LOGGER.info("figure created on %s", figure.canvas.get_window_title())
        return figure

    def show(self, tight_layout=True):
        if tight_layout:
            plt.tight_layout()
        plt.show()

    def save(self, path, **kwargs):
        """Save specified figure to specified path (path can be file object or string path)."""
        # Set figure as current figure.
        plt.figure(self.figure.number)
        # Squeeze things together.
        self.figure.tight_layout()
        plt.savefig(path, **kwargs)


class MplGroupPlotter(MatplotlibPlotter, BaseGroupPlotter, metaclass=abc.ABCMeta):
    """Provides interface for plotting grouped functions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.style_groups = []
        # Plot group line style cycle.
        self.linestyles = ["-", "--", "-.", ":"]
        # Default colour cycle.
        self.default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # Cycles by group. These are created at runtime using the default colour cycle and the
        # lighten_colours() function.
        self._plot_group_colours = {}

    def plot_groups(self, groups):
        for group, functions in groups.items():
            if not functions:
                # Skip empty group.
                continue
            with self._figure_style_context(group):
                # Reset axes colour wheels.
                for axis in self.figure.axes:
                    axis.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
                if self.legend_groups and group not in self.hidden_group_names:
                    # Show group.
                    legend_group = "(%s)" % group
                else:
                    legend_group = None
                self._do_plot(functions, label_suffix=legend_group)
        self._finalise_plot()

    @abc.abstractmethod
    def _do_plot(self, functions, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _finalise_plot(self):
        """Set last plot options such as limits, using the plotted functions."""
        raise NotImplementedError

    def _figure_style_context(self, group):
        """Figure style context manager.

        Used to override the default style for a figure.
        """
        # Find group index.
        if group not in self.style_groups:
            self.style_groups.append(group)
        group_index = self.style_groups.index(group)
        # Get index of linestyle to use (cycles through styles, wrapping back to beginning).
        index = group_index % len(self.linestyles)
        if group not in self._plot_group_colours:
            # Create new cycle with brighter colours.
            cycle = lighten_colours(self.default_color_cycle, 0.5 ** group_index)
            self._plot_group_colours[group] = cycle
        prop_cycler = cycler(color=self._plot_group_colours[group])
        settings = {"lines.linestyle": self.linestyles[index],
                    "axes.prop_cycle": prop_cycler}
        return plt.rc_context(settings)

    def _axis_grayscale_context(self):
        """Sum figure style context manager. This sets the sum colors to greyscale."""
        return plt.rc_context({"axes.prop_cycle": cycler(color=self._grayscale_colours)})

    @property
    def _grayscale_colours(self):
        """Grayscale colour palette."""
        greys = plt.get_cmap('Greys')
        return greys(np.linspace(CONF["plot"]["sum_greyscale_cycle_start"],
                                 CONF["plot"]["sum_greyscale_cycle_stop"],
                                 CONF["plot"]["sum_greyscale_cycle_count"]))


class BodePlotter(MplGroupPlotter):
    def __init__(self, scale_db=True, xlim=None, mag_ylim=None, phase_ylim=None, xlabel=None,
                 ylabel_mag=None, ylabel_phase=None, db_tick_major_step=20, db_tick_minor_step=10,
                 phase_tick_major_step=45, phase_tick_minor_step=15, **kwargs):
        super().__init__(**kwargs)
        self.scale_db = scale_db
        self.xlim = xlim
        self.mag_ylim = mag_ylim
        self.phase_ylim = phase_ylim
        if xlabel is None:
            xlabel = r"$\bf{Frequency}$ (Hz)"
        if ylabel_mag is None:
            if scale_db:
                ylabel_mag = r"$\bf{Magnitude}$ (dB)"
            else:
                ylabel_mag = r"$\bf{Magnitude}$"
        if ylabel_phase is None:
            ylabel_phase = r"$\bf{Phase}$ ($\degree$)"
        self.xlabel = xlabel
        self.ylabel_mag = ylabel_mag
        self.ylabel_phase = ylabel_phase
        self.db_tick_major_step = db_tick_major_step
        self.db_tick_minor_step = db_tick_minor_step
        self.phase_tick_major_step = phase_tick_major_step
        self.phase_tick_minor_step = phase_tick_minor_step

    @property
    def ax1(self):
        return self.figure.axes[0]

    @property
    def ax2(self):
        return self.figure.axes[1]

    def _create_figure(self):
        figure = super()._create_figure()
        # Add magnitude and phase axes.
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212, sharex=ax1)
        # Draw labels etc.
        if self.title is not None:
            # Use ax1 since it's at the top. We could use figure.suptitle but this doesn't
            # behave with tight_layout.
            ax1.set_title(self.title)
        if self.legend:
            ax1.legend(loc=self.legend_loc)
        if self.xlabel is not None:
            ax2.set_xlabel(self.xlabel)
        if self.ylabel_mag is not None:
            ax1.set_ylabel(self.ylabel_mag)
        if self.ylabel_phase is not None:
            ax2.set_ylabel(self.ylabel_phase)

        gridconf = CONF["plot"]["grid"]
        ax1.grid(which="major", alpha=gridconf["alpha_major"], zorder=gridconf["zorder"])
        ax1.grid(which="minor", alpha=gridconf["alpha_minor"], zorder=gridconf["zorder"])
        ax2.grid(which="major", alpha=gridconf["alpha_major"], zorder=gridconf["zorder"])
        ax2.grid(which="minor", alpha=gridconf["alpha_minor"], zorder=gridconf["zorder"])

        # Magnitude and phase tick locators.
        if self.scale_db:
            ax1.yaxis.set_major_locator(MultipleLocator(base=self.db_tick_major_step))
            ax1.yaxis.set_minor_locator(MultipleLocator(base=self.db_tick_minor_step))
        ax2.yaxis.set_major_locator(MultipleLocator(base=self.phase_tick_major_step))
        ax2.yaxis.set_minor_locator(MultipleLocator(base=self.phase_tick_minor_step))
        return figure

    @MplGroupPlotter.figure.setter
    def figure(self, figure):
        if len(figure.axes) != 2:
            raise ValueError("figure must contain two axes")
        self._figure = figure

    def _do_plot(self, functions, **kwargs):
        for response in functions:
            response.draw(self.ax1, self.ax2, scale_db=self.scale_db, **kwargs)

    def plot(self, functions, **kwargs):
        self._do_plot(functions, **kwargs)
        self._finalise_plot()

    def _finalise_plot(self):
        # Update the legend.
        self.ax1.legend(loc=self.legend_loc)

        self._set_limits()

    def _set_limits(self):
        """Set appropriate plot limits, if not explicitly specified."""
        if self.xlim is not None:
            self.ax1.set_xlim(self.xlim)
            self.ax2.set_xlim(self.xlim)

        mag_ylim = self.mag_ylim

        if mag_ylim is None:
            LOGGER.info("No magnitude y-axis limits specified; attempting to use reasonable values.")
            yinterval = self.ax1.yaxis.get_data_interval()
            _, mag_ymargin = self.ax1.margins()
            if self.scale_db:
                # Round up/down to nearest multiple.
                # Note: you might be tempted to turn on axes.autolimit_mode = round_numbers here,
                # which also does this rounding up/down, but also sets the log x-axis limits a full
                # decade lower/above.
                mag_ylim = self._expand_linear_axis_limits(*yinterval, self.db_tick_minor_step,
                                                           mag_ymargin)
            else:
                mag_ylim = self._expand_log_axis_limits(*yinterval, 10, mag_ymargin)

        self.ax1.set_ylim(mag_ylim)

        phase_ylim = self.phase_ylim

        if phase_ylim is None:
            _, phase_ymargin = self.ax2.margins()
            if CONF["plot"]["bode"]["show_full_phase_limits"]:
                LOGGER.info("No phase y-axis limits specified; defaulting to full span due to "
                            "show_full_phase_limits setting.")
                yinterval = (-180, 180)
            else:
                LOGGER.info("No phase y-axis limits specified; attempting to use reasonable values.")
                yinterval = self.ax2.yaxis.get_data_interval()
            phase_ylim = self._expand_linear_axis_limits(*yinterval, self.phase_tick_minor_step,
                                                         phase_ymargin)

        self.ax2.set_ylim(phase_ylim)


class SpectralDensityPlotter(MplGroupPlotter):
    def __init__(self, xlim=None, ylim=None, xlabel=None, ylabel=None, **kwargs):
        super().__init__(**kwargs)
        self.xlim = xlim
        self.ylim = ylim
        if xlabel is None:
            xlabel = r"$\bf{Frequency}$ (Hz)"
        self.xlabel = xlabel
        if ylabel is None:
            ylabel = r"$\bf{Noise}$"
        self.ylabel = ylabel

    @property
    def axis(self):
        return self.figure.axes[0]

    def _create_figure(self):
        figure = super()._create_figure()
        ax = figure.add_subplot(111)
        # Draw labels etc.
        if self.title is not None:
            # Use ax1 since it's at the top. We could use figure.suptitle but this doesn't
            # behave with tight_layout.
            ax.set_title(self.title)
        if self.legend:
            ax.legend(loc=self.legend_loc)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)

        gridconf = CONF["plot"]["grid"]
        ax.grid(which="major", alpha=gridconf["alpha_major"], zorder=gridconf["zorder"])
        ax.grid(which="minor", alpha=gridconf["alpha_minor"], zorder=gridconf["zorder"])

        # Magnitude and phase tick locators.
        return figure

    @MplGroupPlotter.figure.setter
    def figure(self, figure):
        if len(figure.axes) != 1:
            raise ValueError("figure must contain one axis")
        self._figure = figure

    def _do_plot(self, functions, **kwargs):
        for function in functions:
            function.draw(self.axis, **kwargs)

    def plot(self, functions, **kwargs):
        singles = []
        sums = []
        for spectral_density in functions:
            if isinstance(spectral_density, MultiNoiseDensity):
                # Leave to end as we need to set a new prop cycler on the axis.
                sums.append(spectral_density)
            else:
                singles.append(spectral_density)
        self._do_plot(singles, **kwargs)
        with self._axis_grayscale_context():
            self.axis.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
            self._do_plot(sums, **kwargs)
        # Add label to legend.
        self.axis.legend()
        self._finalise_plot()

    def _finalise_plot(self):
        # Update the legend.
        self.axis.legend()

        self._set_limits()

    def _set_limits(self):
        """Set appropriate plot limits, if not explicitly specified."""
        if self.xlim is not None:
            self.axis.set_xlim(self.xlim)

        ylim = self.ylim

        if ylim is None:
            LOGGER.info("No y-axis limits specified; attempting to use reasonable values.")
            yinterval = self.axis.yaxis.get_data_interval()
            _, ymargin = self.axis.margins()
            ylim = self._expand_log_axis_limits(*yinterval, 10, ymargin)

        self.axis.set_ylim(ylim)


class OpAmpGainPlotter(BodePlotter):
    def __init__(self, frequencies=None, fstart=None, fstop=None, npoints=1000,
                 title="Open loop gain"):
        super().__init__(title=title)
        if frequencies is None:
            if any([param is None for param in (fstart, fstop, npoints)]):
                raise ValueError("either frequencies, or all of fstart, fstop and npoints must be "
                                 "specified")
            frequencies = np.logspace(np.log10(Quantity(fstart)), np.log10(Quantity(fstop)),
                                      npoints)
        self.frequencies = np.array(frequencies)

    def response(self, opamp):
        gain = np.array([opamp.gain(frequency) for frequency in self.frequencies])
        series = Series(self.frequencies, gain)
        response = Response(source=opamp.node1, sink=opamp.node3, series=series)
        response.label = opamp.model
        return response

    def plot(self, opamps):
        super().plot([self.response(opamp) for opamp in opamps])

    def show(self):
        plt.show()


class OpAmpNoisePlotter(SpectralDensityPlotter, metaclass=abc.ABCMeta):
    def __init__(self, title, frequencies=None, fstart=None, fstop=None, npoints=1000):
        super().__init__(title=title)
        if frequencies is None:
            if any([param is None for param in (fstart, fstop, npoints)]):
                raise ValueError("either frequencies, or all of fstart, fstop and npoints must be "
                                 "specified")
            frequencies = np.logspace(np.log10(Quantity(fstart)), np.log10(Quantity(fstop)),
                                      npoints)
        self.frequencies = np.array(frequencies)

    @abc.abstractmethod
    def noise(self, opamp):
        raise NotImplementedError

    def plot(self, opamps):
        super().plot([self.noise(opamp) for opamp in opamps])

    def show(self):
        plt.show()


class OpAmpVoltageNoisePlotter(OpAmpNoisePlotter):
    def __init__(self, **kwargs):
        super().__init__(title="Voltage noise", **kwargs)

    def noise(self, opamp):
        series = Series(self.frequencies, opamp.voltage_noise.noise_voltage(self.frequencies))
        spectral_density = NoiseDensity(source=opamp.node3, sink=opamp.node3, series=series)
        spectral_density.label = opamp.model
        return spectral_density


class OpAmpCurrentNoisePlotter(OpAmpNoisePlotter):
    def __init__(self, **kwargs):
        super().__init__(title="Current noise", **kwargs)

    def noise(self, opamp):
        series = Series(self.frequencies, opamp.non_inv_current_noise.noise_current(self.frequencies))
        spectral_density = NoiseDensity(source=opamp.node1, sink=opamp.node1, series=series)
        spectral_density.label = opamp.model
        return spectral_density
