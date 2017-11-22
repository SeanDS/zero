"""Miscellaneous functions"""

import abc
import sys
import math
import numpy as np
import progressbar
from typing import Any, Generator, Dict

class Singleton(abc.ABCMeta):
    """Metaclass implementing the singleton pattern

    This ensures that there is only ever one instance of a class that
    inherits this one.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]

def _n_comb_k(total, choose, repetition=False):
    """Number of combinations of n things taken k at a time

    :param total: total number of things
    :type total: int
    :param choose: number of elements to choose
    :type choose: int
    :param repetition: whether to allow the same values to be repeated in \
                       sequences multiple (up to `choose`) times
    :type repetition: bool
    :return: number of combinations
    :rtype: int
    """

    if repetition:
        return _n_comb_k(total + choose - 1, choose)

    return _binom(total, choose)

def _n_perm_k(total, choose):
    """Number of permutations of n things taken k at a time

    :param total: total number of things
    :type total: int
    :param choose: number of elements to choose
    :type choose: int
    :return: number of permutations
    :rtype: int
    """

    return math.factorial(choose) * _n_comb_k(total, total - choose)

def _binom(total, choose):
    """Calculate binomial coefficient

    From https://stackoverflow.com/a/3025547

    :param total: total number of things
    :type total: int
    :param choose: number of elements to choose
    :type choose: int
    :return: binomial coefficient
    :rtype: int
    """

    if 0 <= choose <= total:
        ntok = 1
        ktok = 1

        for iteration in range(1, min(choose, total - choose) + 1):
            ntok *= total
            ktok *= iteration
            total -= 1
        return ntok // ktok
    else:
        return 0

def _print_progress(sequence, total, update=100000, stream=sys.stdout):
    """Print progress of generator with known length

    :param sequence: sequence to report iteration progress for
    :type sequence: Sequence[Any]
    :param total: number of items generator will produce
    :type total: int
    :param update: number of items to yield before next updating display
    :type update: int
    :param stream: output stream
    :type stream: :class:`io.IOBase`
    :return: input sequence
    :rtype: Generator[Any]
    """

    # set up progress bar
    pbar = progressbar.ProgressBar(widgets=['Calculating: ',
                                            progressbar.Percentage(),
                                            progressbar.Bar(),
                                            progressbar.ETA()],
                                   maxval=100, fd=stream).start()

    count = 0

    for item in sequence:
        count += 1

        if count % update == 0:
            pbar.update(100 * count // total)

        yield item

    # make sure bar finishes at 100
    pbar.update(100)

    # newline before next text
    print(file=stream)

def db(magnitude):
    """Calculate (power) magnitude in decibels

    :param magnitude: magnitude
    :type magnitude: Numeric or :class:`np.array`
    :return: dB magnitude
    :rtype: Numeric or :class:`np.array`
    """

    return 20 * np.log10(magnitude)
