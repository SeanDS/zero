"""Rich display system"""

from importlib import import_module
import numpy as np
import numbers
from tabulate import tabulate

from .config import CircuitConfig
from .components import Resistor, Capacitor, Inductor, OpAmp, Input

CONF = CircuitConfig()

class NodeGraph(object):
    # input shapes per input type
    input_shapes = {"noise": "ellipse", "voltage": "box", "current": "pentagon"}

    def __init__(self, circuit):
        try:
            self.graphviz = import_module("graphviz")
        except ImportError:
            raise NotImplementedError("Node graph representation requires the "
                                      "graphviz package")

        self.circuit = circuit

    def node_graph(self):
        """Create Graphviz node graph"""

        graph = self.graphviz.Digraph(engine=CONF["graphviz"]["engine"])
        graph.attr("node", style=CONF["graphviz"]["node_style"],
                   fontname=CONF["graphviz"]["node_font_name"],
                   fontsize=CONF["graphviz"]["node_font_size"])
        graph.attr("edge", arrowhead=CONF["graphviz"]["edge_arrowhead"])
        graph.attr("graph", splines=CONF["graphviz"]["graph_splines"],
                   label="Made with graphviz and circuit.py",
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

    def _repr_svg_(self):
        """Graphviz rendering for Jupyter notebooks."""
        return self.node_graph()._repr_svg_()

class MatrixDisplay(object):
    def __init__(self, lhs, middle, rhs, headers):
        """Instantiate matrix display"""

        lhs = np.array(lhs)
        middle = np.array(middle)
        rhs = np.array(rhs)
        headers = list(headers)

        if middle.ndim != 2:
            raise ValueError("Middle must have exactly 2 dimensions")

        if lhs.shape[1] + middle.shape[1] + rhs.shape[1] != len(headers):
            raise ValueError("Shape of first dimension of lhs, middle and rhs must "
                             "add up to length of headers")

        self.lhs = lhs
        self.middle = middle
        self.rhs = rhs
        self.headers = headers

    @property
    def matrix(self):
        """Get full matrix as a generator, consisting of left, middle and right sides"""
        # unfortunately can't use `np.concatenate` here, as it changes the full matrix
        # type to whatever is first, which is usually a string

        for l, m, r in zip(self.lhs, self.middle, self.rhs):
            yield self.format_row(l, m, r)

    def format_cell(self, cell):
        if np.iscomplex(cell):
            # convert complex value to magnitude
            cell = np.abs(cell)
        
            # remove imaginary part, which is now zero
            cell = cell.real

        if isinstance(cell, numbers.Number):
            cell = cell.real

            if cell == 0:
                return "---"
            elif abs(cell) == 1:
                # don't format zero or one
                return "%i" % cell
            elif abs(cell) < 1e-12:
                # just show power for tiny stuff
                return "e%d" % round(np.log10(cell))
            
            return "%.2e" % cell
        
        return str(cell)
    
    def format_row(self, left, middle, right):
        yield from [self.format_cell(cell) for cell in left]
        yield from [self.format_cell(cell) for cell in middle]
        yield from [self.format_cell(cell) for cell in right]

    def __repr__(self):
        """Text representation of table"""

        # tabulate data
        return tabulate(self.matrix, self.headers, tablefmt=CONF["format"]["table"])
    
    def _repr_html_(self):
        """HTML table representation"""

        table = "<table>"
        table += "<thead>"
        table += "<tr>"
        table += "".join(["<th>%s</th>" % header_cell for header_cell in self.headers])
        table += "</tr>"
        table += "</thead>"
        table += "<tbody>"
        table += "".join(["<tr>%s</tr>" % "".join(["<td>%s</td>" % self.format_cell(cell) for cell in row]) for row in self.matrix])
        table += "</tbody>"
        table += "</table>"

        return table