"""Rich display system"""

import abc
from importlib import import_module
import collections
import tempfile
import numpy as np
from tabulate import tabulate

from .config import ZeroConfig
from .components import Resistor, Capacitor, Inductor, OpAmp, Input, Component, Node

CONF = ZeroConfig()


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
                   fontsize=CONF["graphviz"]["node_font_size"])
        graph.attr("edge", arrowhead=CONF["graphviz"]["edge_arrowhead"])
        graph.attr("graph", splines=CONF["graphviz"]["graph_splines"],
                   label="Made with graphviz and Zero",
                   fontname=CONF["graphviz"]["graph_font_name"],
                   fontsize=CONF["graphviz"]["graph_font_size"])
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
                attr['label'] = """<<TABLE BORDER="0" BGCOLOR="LightSkyBlue">
                    <TR><TD PORT="plus">+</TD><TD ROWSPAN="3">{0}<BR/>{1}</TD></TR>
                    <TR><TD> </TD></TR>
                    <TR><TD PORT="minus">-</TD></TR>
                </TABLE>>""".format(component.name, component.model)
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
                clause += "%s%s" % (formatted_coefficient,  element_format % element.label())

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
                expression += self.format_coefficient_latex(coefficient) + element_format % element.label()

            # add right hand side, with alignment character
            expression += r"&="
            expression += self.format_exponent_latex(self.format_rhs(rhs_value))

            # add newline
            expression += r"\\"

        expression += r"\end{align}"

        return expression
