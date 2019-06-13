"""Base analysis"""

import sys
import os
import abc
from progressbar import ProgressBar, Percentage, Bar, ETA


class BaseAnalysis(metaclass=abc.ABCMeta):
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
    def __init__(self, circuit, print_progress=False, stream=None):
        if stream is None:
            stream = sys.stdout

        self.circuit = circuit
        self.print_progress = bool(print_progress)
        self.stream = stream

    def progress(self, sequence, total, update=100000):
        """Print progress of generator with known length.

        Parameters
        ----------
        sequence : array_like
            Sequence to report iteration progress for.
        total : int
            The number of items the sequence will contain.
        update : float or int
            The number of items to yield before next updating display.

        Returns
        -------
        array_like
            The input sequence.
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
            # Null file.
            stream = open(os.devnull, "w")

        # Set up progress bar.
        widgets = ['Calculating: ', Percentage(), Bar(), ETA()]
        pbar = ProgressBar(widgets=widgets, max_value=100, fd=stream).start()

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

        pbar.finish()
