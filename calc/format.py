import abc
import math

"""
Formatting functionality for numbers with units
"""

class BaseFormatter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def format(self, value, unit=""):
        raise NotImplemented

class SIFormatter(BaseFormatter):
    """SI unit formatter

    Partially based on https://github.com/ulikoehler/UliEngineering/blob/master/UliEngineering/EngineerIO.py
    """

    unit_prefixes = {-24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
                     -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
                     15: "E", 18: "Z", 21: "Y"}

    @classmethod
    def format(cls, value, unit=""):
        value = float(value)

        # calculate log of value to get exponent
        if value == 0:
            exponent = 0
        else:
            exponent = math.log(abs(value), 10)

        exponent_index = int(math.floor(exponent / 3))

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
