"""Resistor calculations"""

import logging
import itertools
import heapq

from .misc import _n_comb_k, _print_progress
from .format import SIFormatter

class Set(object):
    """Set of resistors and associated operations"""

    # resistor series
    SERIES_E3 = 1
    SERIES_E6 = 2
    SERIES_E12 = 3
    SERIES_E24 = 4
    SERIES_E48 = 5
    SERIES_E96 = 6
    SERIES_E192 = 7
    SERIES_CUSTOM = 8

    # base values for E192 (from which E96 and E48 are derived)
    VALUES_E192 = [
        1.00, 1.01, 1.02, 1.04, 1.05, 1.06, 1.07, 1.09, 1.10, 1.11, 1.13, 1.14,
        1.15, 1.17, 1.18, 1.20, 1.21, 1.23, 1.24, 1.26, 1.27, 1.29, 1.30, 1.32,
        1.33, 1.35, 1.37, 1.38, 1.40, 1.42, 1.43, 1.45, 1.47, 1.49, 1.50, 1.52,
        1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.65, 1.67, 1.69, 1.72, 1.74, 1.76,
        1.78, 1.80, 1.82, 1.84, 1.87, 1.89, 1.91, 1.93, 1.96, 1.98, 2.00, 2.03,
        2.05, 2.08, 2.10, 2.13, 2.15, 2.18, 2.21, 2.23, 2.26, 2.29, 2.32, 2.34,
        2.37, 2.40, 2.43, 2.46, 2.49, 2.52, 2.55, 2.58, 2.61, 2.64, 2.67, 2.71,
        2.74, 2.77, 2.80, 2.84, 2.87, 2.91, 2.94, 2.98, 3.01, 3.05, 3.09, 3.12,
        3.16, 3.20, 3.24, 3.28, 3.32, 3.36, 3.40, 3.44, 3.48, 3.52, 3.57, 3.61,
        3.65, 3.70, 3.74, 3.79, 3.83, 3.88, 3.92, 3.97, 4.02, 4.07, 4.12, 4.17,
        4.22, 4.27, 4.32, 4.37, 4.42, 4.48, 4.53, 4.59, 4.64, 4.70, 4.75, 4.81,
        4.87, 4.93, 4.99, 5.05, 5.11, 5.17, 5.23, 5.30, 5.36, 5.42, 5.49, 5.56,
        5.62, 5.69, 5.76, 5.83, 5.90, 5.97, 6.04, 6.12, 6.19, 6.26, 6.34, 6.42,
        6.49, 6.57, 6.65, 6.73, 6.81, 6.90, 6.98, 7.06, 7.15, 7.23, 7.32, 7.41,
        7.50, 7.59, 7.68, 7.77, 7.87, 7.96, 8.06, 8.16, 8.25, 8.35, 8.45, 8.56,
        8.66, 8.76, 8.87, 8.98, 9.09, 9.20, 9.31, 9.42, 9.53, 9.65, 9.76, 9.88]

    # base values for E24 (from which E12, E6 and E3 are derived)
    VALUES_E24 = [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
        3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
    ]

    def __init__(self, series=None, tolerance=None, max_exp=6, min_exp=0,
                 max_series=1, min_series=1, max_parallel=1, min_parallel=1):
        """Initialise resistor set

        :param series: resistor series
        :param tolerance: resistor tolerance
        :param max_exp: maximum exponent
        :param min_exp: minimum exponent
        :param max_series: maximum number of series combinations
        :param min_series: minimum number of series combinations
        :param max_parallel: maximum number of parallel combinations
        :param min_parallel: minimum number of parallel combinations
        """

        if series is None:
            logging.getLogger("resistor").info("Using E24 series by default")
            series = self.SERIES_E24
        else:
            str_series = str(series).lower()

            if str_series == "e3":
                series = self.SERIES_E3
            elif str_series == "e6":
                series = self.SERIES_E6
            elif str_series == "e12":
                series = self.SERIES_E12
            elif str_series == "e24":
                series = self.SERIES_E24
            elif str_series == "e48":
                series = self.SERIES_E48
            elif str_series == "e96":
                series = self.SERIES_E96
            elif str_series == "e192":
                series = self.SERIES_E192

        if tolerance is None:
            logging.getLogger("resistor").info("Using gold (5%) tolerance by default")
            tolerance = Resistor.TOL_GOLD
        else:
            # format tolerance
            tolerance = float(str(tolerance).strip().replace("%", ""))

        self.series = int(series)
        self.tolerance = float(tolerance)
        self.max_exp = int(max_exp)
        self.min_exp = int(min_exp)
        self.max_series = int(max_series)
        self.min_series = int(min_series)
        self.max_parallel = int(max_parallel)
        self.min_parallel = int(min_parallel)

    def _base_numbers(self):
        """Get set's numbers between 1 and 10"""

        if self.series is self.SERIES_E192:
            return self.VALUES_E192
        elif self.series is self.SERIES_E96:
            return self.VALUES_E192[::2]
        elif self.series is self.SERIES_E48:
            return self.VALUES_E192[::4]
        elif self.series is self.SERIES_E24:
            return self.VALUES_E24
        elif self.series is self.SERIES_E12:
            return self.VALUES_E24[::2]
        elif self.series is self.SERIES_E6:
            return self.VALUES_E24[::4]
        elif self.series is self.SERIES_E3:
            return self.VALUES_E24[::8]
        else:
            raise ValueError("Unrecognised resistor series")

    def _base_resistors(self):
        """Calculate set's base resistors"""

        if self.min_exp > self.max_exp:
            raise ValueError("max_exp must be >= min_exp")

        # calculate exponents of the base numbers and fill values set
        for exp in range(self.min_exp, self.max_exp + 1):
            for base in self._base_numbers():
                yield Resistor(base * 10 ** exp, tolerance=self.tolerance)

    def combinations(self):
        """Get series/parallel resistor combinations

        This returns a generator which yields non-unique resistor values or
        series/parallel combinations of values.
        """

        if self.min_series < 2 or self.min_parallel < 2:
            # yield single resistors
            yield from self._base_resistors()

        # compute series and parallel combinations
        yield from self._series_combinations()
        yield from self._parallel_combinations()

    def _series_combinations(self):
        """Returns series combinations of the specified set of values"""

        # create series collections
        for resistors in self._value_combinations(self._base_resistors(),
                                                  self.min_series,
                                                  self.max_series):
            yield Collection(resistors, vtype=Collection.TYPE_SERIES)

    def _parallel_combinations(self):
        """Returns parallel combinations of the specified set of values"""

        # create series collections
        for resistors in self._value_combinations(self._base_resistors(),
                                                  self.min_parallel,
                                                  self.max_parallel):
            yield Collection(resistors, vtype=Collection.TYPE_PARALLEL)

    @staticmethod
    def _value_combinations(values, min_count, max_count):
        """Returns combinations of the specified set of values between \
        min_count and max_count

        :param values: set of values
        :param min_count: minimum number of values in each combination
        :param max_count: maximum number of values in each combination
        """

        values = list(values)

        min_count = int(min_count)
        max_count = int(max_count)

        if min_count > max_count:
            raise ValueError("max_count must be >= min_count")
        elif max_count < 2:
            return []

        for count in range(min_count, max_count + 1):
            yield from itertools.combinations_with_replacement(values, count)

    def n_combinations(self):
        """Get number of possible combinations with the current settings"""

        count = 0

        # number of base resistors
        n_base_resistors = len(list(self._base_resistors()))

        # first count the base resistors
        if self.min_series < 2 or self.min_parallel < 2:
            count += n_base_resistors

        # add on series resistors
        if self.max_series >= 2:
            for i in range(self.min_series, self.max_series + 1):
                count += _n_comb_k(n_base_resistors, i, repetition=True)

        # add on parallel resistors
        if self.max_parallel >= 2:
            for i in range(self.min_parallel, self.max_parallel + 1):
                count += _n_comb_k(n_base_resistors, i, repetition=True)

        return count

    def closest(self, resistance, n_values=3, progress=True):
        """Returns closest resistors in set to target resistance

        :param resistance: target resistance
        :param n_values: number of resistors to match
        :param progress: show interactive progress bar
        """

        resistance = float(resistance)
        n_values = int(n_values)

        if resistance < 0:
            raise ValueError("Resistance must be > 0")

        # calculate number of results
        n_combinations = self.n_combinations()
        logging.getLogger("resistor").info("Calculating %i combinations",
                                            n_combinations)

        # generate combinations
        combinations = self.combinations()

        if progress:
            # add progress bar between voltage and heapq generators
            combinations = _print_progress(combinations, n_combinations)

        logging.getLogger("resistor").debug("Finding closest resistor matches")
        return heapq.nsmallest(n_values, combinations,
                               key=lambda i: abs(i.resistance - resistance))

class Resistor(object):
    """Represents a resistor or set of series or parallel resistors"""

    # tolerances, in percent (+/-)
    TOL_GREY = 0.05
    TOL_VIOLET = 0.1
    TOL_BLUE = 0.25
    TOL_GREEN = 0.5
    TOL_BROWN = 1.0
    TOL_RED = 2.0
    TOL_GOLD = 5.0
    TOL_SILVER = 10.0
    TOL_NONE = 20.0

    def __init__(self, value=None, tolerance=None):
        """Instantiate a resistor

        :param value: resistor value
        :param tolerance: optional resistor tolerance
        """

        # default properties
        self._resistance = None
        self._tolerance = None

        if value is not None:
            self.resistance = float(value)

        if tolerance is None:
            # default tolerance
            tolerance = self.TOL_NONE

        self.tolerance = float(tolerance)

    @property
    def resistance(self):
        """Get resistance in ohms"""

        return self._resistance

    @resistance.setter
    def resistance(self, resistance):
        """Set resistance

        :param resistance: resistance, in ohms
        """

        self._resistance = float(resistance)

    @property
    def tolerance(self):
        """Get tolerance in percent"""

        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        """Set tolerance

        :param tolerance: tolerance, in percent
        """

        self._tolerance = float(tolerance)

    @property
    def abs_tolerance(self):
        """Absolute tolerance"""
        return self.resistance * self.tolerance / 100

    @property
    def abs_inv_tolerance(self):
        """Absolute inverse tolerance"""
        return (1 / self.resistance) * self.tolerance / 100

    def label(self, tolerance=True):
        """Label for this resistor

        :param tolerance: show tolerances
        """

        label = SIFormatter.format(self.resistance, "Ω")

        if tolerance:
            label += " ± {}%".format(SIFormatter.format(self.tolerance))

        return label

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.label()

class Collection(Resistor):
    """Represents a collection of resistors that behave like a single resistor"""

    # configuration types
    TYPE_SERIES = 1
    TYPE_PARALLEL = 2

    def __init__(self, resistors, vtype=None):
        """Instantiate a resistor collection

        :param resistors: iterable containing :class:`Resistor` objects
        :param vtype: configuration type (series or parallel)
        """

        # call parent
        super(Collection, self).__init__()

        # default to series
        if vtype is None:
            vtype = self.TYPE_SERIES
        elif vtype not in (self.TYPE_SERIES, self.TYPE_PARALLEL):
            raise ValueError("Unrecognised vtype")

        self.resistors = list(resistors)
        self.vtype = vtype

    @Resistor.resistance.getter
    def resistance(self):
        """Calculate resistance in ohms"""

        if self.vtype is self.TYPE_SERIES:
            # series sum
            return sum(resistor.resistance for resistor in self.resistors)
        else:
            # parallel sum
            return 1 / sum(1 / resistor.resistance for resistor in self.resistors)

    @Resistor.tolerance.getter
    def tolerance(self):
        """Tolerance of combination in percent

        Important note: this assumes that that resistors are independently
        random variables. Resistors from the same batch or line will not be
        fully independent. Manufacturers may also select better toreranced
        resistors for other lines, resulting in a non-normal distribution, which
        makes this calculation inaccurate.

        Furthermore, the mean and variance of an inverse distribution - which is
        used for the parallel calculation - is not generally the inverse of the
        original mean and variance. A more accurate calculation would involve a
        Taylor expansion, which we don't do here (see
        http://paulorenato.com/index.php/109).
        """

        if self.vtype is self.TYPE_SERIES:
            # quadrature sum of absolute tolerances
            abs_q_sum = sum([resistor.abs_tolerance ** 2
                             for resistor in self.resistors]) ** 0.5

            # express as percentage of total resistance
            return 100 * abs_q_sum / self.resistance
        else:
            # quadrature sum of absolute inverse tolerances
            abs_q_sum = sum([resistor.abs_inv_tolerance ** 2
                             for resistor in self.resistors]) ** 0.5

            # express as percentage of total inverse resistance
            return 100 * abs_q_sum / sum([1 / resistor.resistance
                                          for resistor in self.resistors])

    def series_equivalent(self):
        """Return collection with identical resistors in series"""

        return Collection(self.resistors, self.TYPE_SERIES)

    def parallel_equivalent(self):
        """Return collection with identical resistors in parallel"""

        return Collection(self.resistors, self.TYPE_PARALLEL)

    def label(self, constituents=True, *args, **kwargs):
        """Label for this collection

        :param constituents: show constituent resistors and tolerances
        """

        label = super(Collection, self).label(*args, **kwargs)

        if constituents:
            if self.vtype is self.TYPE_SERIES:
                combined_resistances = " + ".join([r.label(*args, **kwargs)
                                                   for r in self.resistors])
            else:
                combined_resistances = " || ".join([r.label(*args, **kwargs)
                                                    for r in self.resistors])

            label += " ({})".format(combined_resistances)

        return label

    def __str__(self):
        return self.label()
