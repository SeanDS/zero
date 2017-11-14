"""Formatting functionality for numbers with units"""

import abc
import math
import re
from typing import Any

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
    UNIT_PREFICES = {-24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
                     -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
                     15: "E", 18: "Z", 21: "Y"}

    # other strings used to represent prefices
    PREFIX_ALIASES = {"u": "µ"}

    # regular expression to find values with unit prefixes in text
    # this technically allows strings with both exponents and unit prefices,
    # like ".1e-6.M", but these should fail later validation
    VALUE_REGEX = re.compile("^([+-]?\d*\.?\d*)([eE]([+-]?\d*\.?\d*))?([\w])?")

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
        if not (min(cls.UNIT_PREFICES.keys())
                < exponent_index
                < max(cls.UNIT_PREFICES.keys())):
            raise ValueError("{} out of range".format(value))

        # create suffix with unit
        suffix = cls.UNIT_PREFICES[exponent_index * 3] + unit

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

    @classmethod
    def prefices(cls):
        yield from cls.UNIT_PREFICES.values()
        yield from cls.PREFIX_ALIASES.keys()

    @classmethod
    def parse(cls, value_str: Any) -> float:
        if isinstance(value_str, (int, float)):
            return float(value_str)

        # find floating point numbers and optional unit prefix
        results = cls.VALUE_REGEX.findall(value_str)[0]

        # first result is the base number
        base = float(results[0])

        if results[1] != '' and results[3] != '':
            raise Exception("Cannot specify both exponent and unit prefix")

        # handle exponent
        if results[1] != '':
            # prefix specified
            exponent = float(results[2])
        elif results[3] != '':
            exponent = cls.unit_exponent(results[3])
        else:
            exponent = 0

        # return float equivalent
        return base * 10 ** exponent

    @classmethod
    def unit_exponent(cls, prefix: str) -> int:
        if prefix in cls.PREFIX_ALIASES:
            prefix = cls.PREFIX_ALIASES[prefix]

        for exponent, this_prefix in cls.UNIT_PREFICES.items():
            if this_prefix == prefix:
                return exponent
