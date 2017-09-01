import logging
import itertools

"""
TODO: add min_ and max_series parameters
"""

class Set(object):
    E3     = 1
    E6     = 2
    E12    = 3
    E24    = 4
    E48    = 5
    E96    = 6
    E192   = 7
    CUSTOM = 8

    SERIES_E192 = [
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

    SERIES_E24 = [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
        3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
    ]

    def __init__(self, series=None, tolerance=None):
        if series is None:
            logging.getLogger("resistor").info("Using E24 series by default")
            series = self.E24
        else:
            series = str(series).lower()

            if series == "e3":
                series = self.E3
            elif series == "e6":
                series = self.E6
            elif series == "e12":
                series = self.E12
            elif series == "e24":
                series = self.E24
            elif series == "e48":
                series = self.E48
            elif series == "e96":
                series = self.E96
            elif series == "e192":
                series = self.E192

        if tolerance is None:
            logging.getLogger("resistor").info("Using gold (5%) tolerance by default")
            tolerance = Resistor.TOL_GOLD

        self.series = int(series)
        self.tolerance = float(tolerance)

    def _get_base_numbers(self):
        if self.series is self.E192:
            return self.SERIES_E192
        elif self.series is self.E96:
            return self.SERIES_E192[::2]
        elif self.series is self.E48:
            return self.SERIES_E192[::4]
        elif self.series is self.E24:
            return self.SERIES_E24
        elif self.series is self.E12:
            return self.SERIES_E24[::2]
        elif self.series is self.E6:
            return self.SERIES_E24[::4]
        elif self.series is self.E3:
            return self.SERIES_E24[::8]
        else:
            raise ValueError("Unrecognised resistor series")

    def combinations(self, max_exp=6, min_exp=0, max_series=1, min_series=2,
                     max_parallel=1, min_parallel=2):
        """Get series/parallel resistor combinations

        This returns a generator which yields non-unique resistor values or
        series/parallel combinations of values. To obtain a unique set of
        values, use :method:`~unique_combinations`.

        :param max_exp: maximum exponent
        :param min_exp: minimum exponent
        :param max_series: maximum number of series combinations
        :param min_series: minimum number of series combinations
        :param max_parallel: maximum number of parallel combinations
        :param min_parallel: minimum number of parallel combinations
        """

        max_exp = int(max_exp)
        min_exp = int(min_exp)

        # base resistor numbers between 1 and 10 ohms
        base_numbers = self._get_base_numbers()

        # list to store base resistor values from which combinations are
        # computed
        values = []

        # calculate exponents of the base numbers and fill values set
        for exp in range(min_exp, max_exp + 1):
            values.extend([Resistor(v * 10 ** exp, tolerance=self.tolerance)
                           for v in base_numbers])

        # yield single resistors
        yield from values

        # compute series and parallel combinations
        yield from self.series_combinations(values, max_series, min_series)
        yield from self.parallel_combinations(values, max_parallel, min_parallel)

    def unique_combinations(self, *args, **kwargs):
        """Guaranteed unique resistor combinations

        This method returns a set of resistor combinations instead of the
        generator used by :method:`~combinations`. It is more memory intensive
        but guarantees unique combinations.
        """

        return set(self.combinations(*args, **kwargs))

    def series_combinations(self, values, max_series=2, min_series=2):
        """Returns series combinations of the specified set of values

        :param values: set of values
        :param max_series: maximum number of series resistors
        :param min_series: minimum number of series resistors
        """

        # create series collections
        for resistors in self._value_combinations(values, max_series,
                                                  min_series):
            yield Collection(resistors, vtype=Collection.TYPE_SERIES)

    def parallel_combinations(self, values, max_parallel=2, min_parallel=2):
        """Returns parallel combinations of the specified set of values

        :param values: set of values
        :param max_parallel: maximum number of parallel resistors
        :param min_parallel: minimum number of parallel resistors
        """

        # create series collections
        for resistors in self._value_combinations(values, max_parallel,
                                                  min_parallel):
            yield Collection(resistors, vtype=Collection.TYPE_PARALLEL)

    def _value_combinations(self, values, max_count=2, min_count=2):
        """Returns combinations of the specified set of values between \
        min_count and max_count

        :param values: set of values
        :param max_count: maximum number of values in each combination
        :param min_count: minimum number of values in each combination
        """

        values = list(values)

        max_count = int(max_count)
        min_count = int(min_count)

        if max_count < 2:
            return []
        elif min_count < 2:
            raise ValueError("min_count must be >= 2")
        elif min_count > max_count:
            raise ValueError("max_count must be >= min_count")

        for n in range(min_count, max_count + 1):
            yield from itertools.combinations_with_replacement(values, n)

class Resistor(object):
    # tolerances, in percent (+/-)
    TOL_GREY = 0.05
    TOL_VIOLET = 0.1
    TOL_BLUE = 0.25
    TOL_GREEN = 0.5
    TOL_BROWN = 1
    TOL_RED = 2
    TOL_GOLD = 5
    TOL_SILVER = 10
    TOL_NONE = 20

    def __init__(self, value, tolerance=None):
        if tolerance is None:
            tolerance = self.TOL_NONE

        self.resistance = float(value)
        self.tolerance = float(tolerance)

    @property
    def resistance(self):
        return self._resistance

    @resistance.setter
    def resistance(self, resistance):
        self._resistance = float(resistance)

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = float(tolerance)

    @property
    def abs_tolerance(self):
        """Absolute tolerance"""
        return self.resistance * self.tolerance / 100

    @property
    def abs_inv_tolerance(self):
        """Absolute inverse tolerance"""
        return (1 / self.resistance) * self.tolerance / 100

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "{0}±{1}%".format(self.resistance, self.tolerance)

class Collection(Resistor):
    TYPE_SERIES = 1
    TYPE_PARALLEL = 2

    def __init__(self, resistors, vtype=None):
        if vtype is None:
            vtype = self.TYPE_SERIES

        if vtype not in (self.TYPE_SERIES, self.TYPE_PARALLEL):
            raise ValueError("Unrecognised vtype")

        self.resistors = list(resistors)
        self.vtype = vtype

    @property
    def resistance(self):
        if self.vtype is self.TYPE_SERIES:
            # series sum
            return sum(resistor.resistance for resistor in self.resistors)
        else:
            # parallel sum
            return 1 / sum(1 / resistor.resistance for resistor in self.resistors)

    @resistance.setter
    def resistance(self, resistance):
        return NotImplemented

    @property
    def tolerance(self):
        """Tolerance of combination

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

    @tolerance.setter
    def tolerance(self, tolerance):
        return NotImplemented

    def __str__(self):
        if self.vtype is self.TYPE_SERIES:
            combined_resistances = " + ".join([str(r) for r in self.resistors])
        else:
            combined_resistances = " || ".join([str(r) for r in self.resistors])

        return "{0}±{1}% ({2})".format(self.resistance, self.tolerance,
                                       combined_resistances)
