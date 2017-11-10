"""Formatting functionality for numbers with units"""

import abc
import math

class BaseFormatter(metaclass=abc.ABCMeta):
    """Abstract class for all formatters"""

    @abc.abstractclassmethod
    def format(cls, value, unit=""):
        """Method to format the specified value and unit

        :param value: value to format
        :unit: optional unit to append
        """

        raise NotImplementedError

    @staticmethod
    def exponent(value):
        """Calculate the exponent of 10 corresponding to the specified value

        :param value: value to find exponent of 10 for
        """

        if value == 0:
            return 0

        # calculate log of value to get exponent
        return math.log(abs(value), 10)

class SIFormatter(BaseFormatter):
    """SI unit formatter

    Partially based on `EngineerIO` from
    https://github.com/ulikoehler/UliEngineering/.
    """

    # exponents and their SI prefixes
    unit_prefixes = {-24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
                     -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
                     15: "E", 18: "Z", 21: "Y"}

    @classmethod
    def format(cls, value, unit=""):
        """Method to format the specified value and unit

        :param value: value to format
        :unit: optional unit to append
        """

        value = float(value)

        # multiple of 3 corresponding to the unit prefix list
        exponent_index = int(math.floor(cls.exponent(value) / 3))

        # check for out of range exponents
        if not (min(cls.unit_prefixes.keys())
                < exponent_index
                < max(cls.unit_prefixes.keys())):
            raise ValueError("{} out of range".format(value))

        # create suffix with unit
        suffix = cls.unit_prefixes[exponent_index * 3] + unit

        # numerals without scale; should always be < 1000 (10 ** 3)
        numerals = value * (10 ** -(exponent_index * 3))

        # show only three digits
        if numerals < 10:
            result = "{:.2f}".format(numerals)
        elif numerals < 100:
            result = "{:.1f}".format(numerals)
        else:
            result = str(int(round(numerals)))

        # return formatted string with space between numeral and unit if present
        return "{0} {1}".format(result, suffix) if suffix else result
