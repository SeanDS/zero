import sys
import os
import abc
import copy
import statistics
import progressbar

from ..components import (Component, Resistor, Capacitor, Inductor, OpAmp,
                          Input, Node, ComponentNoise, NodeNoise)

class BaseAnalysis(object, metaclass=abc.ABCMeta):
    """Base class for circuit analysis.

    Parameters
    ----------
    circuit : :class:`.Circuit`
        The circuit to analyse.
    print_progress : :class:`bool`, optional
        Whether to print analysis output.
    stream : :class:`io.IOBase`, optional
        Stream to print analysis output to.
    """

    def __init__(self, circuit, print_progress=False, stream=sys.stdout):
        self.circuit = circuit
        self.print_progress = bool(print_progress)
        self.stream = stream

    def component_index(self, component):
        """Get component serial number.

        Parameters
        ----------
        component : :class:`~.Component`
            component

        Returns
        -------
        :class:`int`
            component serial number

        Raises
        ------
        ValueError
            if component not found
        """

        return self.circuit.components.index(component)

    def node_index(self, node):
        """Get node serial number.

        This does not include the ground node, so the first non-ground node
        has serial number 0.

        Parameters
        ----------
        node : :class:`~.Node`
            node

        Returns
        -------
        :class:`int`
            node serial number

        Raises
        ------
        ValueError
            if ground node is specified or specified node is not found
        """

        if node == Node("gnd"):
            raise ValueError("ground node does not have an index")

        return list(self.circuit.non_gnd_nodes).index(node)

    @property
    def elements(self):
        """Matrix elements.

        Returns a sequence of elements - either components or nodes - in the
        order in which they appear in the matrix

        Yields
        ------
        :class:`~.components.Component`, :class:`~.components.Node`
            matrix elements
        """

        yield from self.circuit.components
        yield from self.circuit.non_gnd_nodes

    @property
    def element_names(self):
        """Names of elements (components and nodes) within the circuit.

        Yields
        ------
        :class:`str`
            matrix element names
        """

        return [element.name for element in self.elements]

    @property
    def mean_resistance(self):
        """Average circuit resistance"""
        return statistics.mean([resistor.resistance for resistor in self.circuit.resistors])

    def progress(self, sequence, total, update=100000):
        """Print progress of generator with known length

        :param sequence: sequence to report iteration progress for
        :type sequence: Sequence[Any]
        :param total: number of items generator will produce
        :type total: int
        :param update: number of items to yield before next updating display
        :type update: float or int
        :return: input sequence
        :rtype: Generator[Any]
        """

        total = int(total)
        update = float(update)

        if total <= 0:
            raise ValueError("total must be > 0")

        if update <= 0:
            raise ValueError("update must be > 0")

        if self.print_progress:
            stream = self.stream
        else:
            # null file
            stream = open(os.devnull, "w")

        # set up progress bar
        pbar = progressbar.ProgressBar(widgets=['Calculating: ',
                                                progressbar.Percentage(),
                                                progressbar.Bar(),
                                                progressbar.ETA()],
                                       max_value=100, fd=stream).start()

        count = 0

        for item in sequence:
            count += 1

            if count % update == 0:
                if count == total:
                    fraction = 1
                else:
                    fraction = 100 * count // total
                
                pbar.update(fraction)

            yield item
