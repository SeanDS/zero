"""Miscellaneous functions"""

import abc
import sys
import math
import numpy as np
import progressbar

class Singleton(abc.ABCMeta):
    """Metaclass implementing the singleton pattern

    This ensures that there is only ever one instance of a class that
    inherits this one.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by class
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]

class NamedInstance(abc.ABCMeta):
    """Metaclass to implement a single named instance pattern

    This ensures that there is only ever one instance of a class with a specific
    name, as provided by the "name" constructor argument.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by name
    _names = {}

    def __call__(cls, name, *args, **kwargs):
        name = name.lower()

        if name not in cls._names:
            cls._names[name] = super(NamedInstance, cls).__call__(name, *args, **kwargs)

        return cls._names[name]

def _print_progress(sequence, total, update=100000, stream=sys.stdout):
    """Print progress of generator with known length

    :param sequence: sequence to report iteration progress for
    :type sequence: Sequence[Any]
    :param total: number of items generator will produce
    :type total: int
    :param update: number of items to yield before next updating display
    :type update: float or int
    :param stream: output stream
    :type stream: :class:`io.IOBase`
    :return: input sequence
    :rtype: Generator[Any]
    """

    total = int(total)
    update = float(update)

    if total <= 0:
        raise ValueError("total must be > 0")

    if update <= 0:
        raise ValueError("update must be > 0")

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
            stream.flush()

        yield item

def db(magnitude):
    """Calculate (power) magnitude in decibels

    :param magnitude: magnitude
    :type magnitude: Numeric or :class:`np.array`
    :return: dB magnitude
    :rtype: Numeric or :class:`np.array`
    """

    return 20 * np.log10(magnitude)