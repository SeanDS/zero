"""Miscellaneous functions"""

import math
import numpy as np
import progressbar

class Singleton(type):
    """Metaclass implementing the singleton pattern"""

    # list of children
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]

def _n_comb_k(total, choose, repetition=False):
    """Number of combinations of n things taken k at a time

    :param total: total number of things
    :param choose: number of elements to choose
    :param repetition: whether to allow the same values to be repeated in \
                       sequences multiple (up to `choose`) times
    """

    if repetition:
        return _n_comb_k(total + choose - 1, choose)

    return _binom(total, choose)

def _n_perm_k(total, choose):
    """Number of permutations of n things taken k at a time

    :param total: total number of things
    :param choose: number of elements to choose
    """

    return math.factorial(choose) * _n_comb_k(total, total - choose)

def _binom(total, choose):
    """Binomial coefficient

    From https://stackoverflow.com/a/3025547

    :param total: total number of things
    :param choose: number of elements to choose
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

def _print_progress(generator, total, update=100000):
    """Print progress of generator with known length

    :param generator: generator to print progress for
    :param total: number of items generator will produce
    :param update: number of items to yield before next updating display
    """

    # set up progress bar
    pbar = progressbar.ProgressBar(widgets=['Calculating: ',
                                            progressbar.Percentage(),
                                            progressbar.Bar(),
                                            progressbar.ETA()],
                                   maxval=100).start()

    count = 0

    for item in generator:
        count += 1

        if count % update == 0:
            pbar.update(100 * count // total)

        yield item

    # make sure bar finishes at 100
    pbar.update(100)

    # newline before next text
    print()

def db(magnitude):
    return 20 * np.log10(magnitude)
