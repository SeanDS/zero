"""Circuit node graph display.

Sean Leavey
"""

from zero import Circuit
from zero.display import NodeGraph

if __name__ == "__main__":
    # Create circuit object.
    circuit = Circuit()

    # Add components.
    circuit.add_capacitor(value="10u", node1="gnd", node2="n1")
    circuit.add_resistor(value="430", node1="n1", node2="nm")
    circuit.add_resistor(value="43k", node1="nm", node2="nout")
    circuit.add_capacitor(value="47p", node1="nm", node2="nout")
    circuit.add_library_opamp(model="LT1124", node1="gnd", node2="nm", node3="nout")

    # Show.
    graph = NodeGraph(circuit)

    def in_notebook():
        """Detect if we're inside an IPython/Jupyter notebook."""
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except (ImportError, AttributeError):
            return False
        return True

    if in_notebook():
        from IPython.display import display
        display(graph)
    else:
        graph.view_pdf()
