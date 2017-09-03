import math
import progressbar

def _n_comb_k(n, k, repetition=False):
    """Number of combinations of n things taken k at a time

    :param n: number of things
    :param k: number of elements chosen
    :param repetition: whether to allow the same values to be repeated in \
                       sequences multiple (up to k) times
    """

    if repetition:
        return _n_comb_k(n + k - 1, k)

    return _binom(n, k)

def _n_perm_k(n, k):
    """Number of permutations of n things taken k at a time

    :param n: number of things
    :param k: number of elements chosen
    """

    return math.factorial(k) * _n_comb_k(n, n - k)

def _binom(n, k):
    """Binomial coefficient

    From https://stackoverflow.com/a/3025547

    :param n: number of things
    :param k: number of elements chosen
    """

    if 0 <= k <= n:
        ntok = 1
        ktok = 1

        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def _print_progress(generator, n):
    """Print progress of generator with known length

    :param generator: generator to print progress for
    :n: number of items generator will produce
    """

    # set up progress bar
    pbar = progressbar.ProgressBar(widgets=['Calculating: ',
                                            progressbar.Percentage(),
                                            progressbar.Bar(),
                                            progressbar.ETA()],
                                   maxval=100).start()

    i = 0

    for item in generator:
        i += 1

        if i % 100000 == 0:
            pbar.update(100 * i // n)

        yield item

    # make sure bar finishes at 100
    pbar.update(100)

    # newline before next text
    print()
