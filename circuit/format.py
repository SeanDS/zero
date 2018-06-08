"""Formatting functionality for numbers with units"""

import abc
import math
import re
from decimal import Decimal

class BaseFormatter(metaclass=abc.ABCMeta):
    """Abstract class for all formatters"""

    @abc.abstractclassmethod
    def format(cls, value, unit=""):
        """Method to format the specified value and unit

        :param value: value to format
        :type value: Numeric
        :param unit: optional unit to append
        :type unit: str
        :return: formatted value and unit
        :rtype: str
        """

        raise NotImplementedError

    @staticmethod
    def exponent(value):
        """Calculate the exponent of 10 corresponding to the specified value

        :param value: value to find exponent of 10 for
        :type value: Numeric
        :return: exponent of 10 for the given value
        :rtype: float
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

    # supported units
    SI_UNITS = ["m", "kg", "s", "A", "K", "mol", "cd", # base units
                "rad", "sr", "Hz", "N", "Pa", "J", "W", # derived units
                "C", "V", "F", "Ω", "S", "Wb", "T", "H",
                "°C", "lm", "lx", "Bq", "Gy", "Sv", "kat"]

    # exponents and their SI prefixes
    UNIT_PREFICES = {-24: "y", -21: "z", -18: "a", -15: "f", -12: "p", -9: "n",
                     -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T",
                     15: "E", 18: "Z", 21: "Y"}

    # other strings used to represent prefices
    PREFIX_ALIASES = {"u": "µ"}

    # regular expression to find values with unit prefixes and units in text
    VALUE_REGEX_STR = (r"^([+-]?\d*\.?\d*)" # base
                       r"([eE]([+-]?\d*\.?\d*))?\s*" # numeric exponent
                       r"([yzafpnuµmkMGTEZY])?" # unit prefix exponent
                       r"(m|kg|s|A|K|mol|cd|rad|" # SI unit
                       r"sr|Hz|N|Pa|J|W|C|V|F|Ω|"
                       r"S|Wb|T|H|°C|lm|lx|Bq|Gy"
                       r"|Sv|kat)?")
    VALUE_REGEX = re.compile(VALUE_REGEX_STR)

    @classmethod
    def format(cls, value, unit=""):
        """Method to format the specified value and unit

        :param value: value to format
        :type value: Numeric
        :param unit: optional unit to append
        :type unit: str
        :return: formatted value and unit
        :rtype: str
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
        """Get unit prefices, including aliases

        :return: unit prefices/aliases
        :rtype: Generator[str]
        """

        yield from cls.UNIT_PREFICES.values()
        yield from cls.PREFIX_ALIASES.keys()

    @classmethod
    def parse(cls, value_str):
        """Parse value string as number

        :param value_str: value to parse
        :type value_str: str
        :return: parsed number, without units or prefix
        :rtype: float
        """

        # don't need to handle units if there aren't any
        if isinstance(value_str, (int, float)):
            return float(value_str), None

        # find floating point numbers and optional unit prefix in string
        results = re.match(cls.VALUE_REGEX, value_str)

        # first result should be the base number
        base = Decimal(results.group(1))

        # handle exponent
        if results.group(3) or results.group(4):
            exponent = 0

            if results.group(3):
                # exponent specified directly
                exponent += float(results.group(3))
            
            if results.group(4):
                # exponent specified as unit prefix
                exponent += cls.unit_exponent(results.group(4))
        else:
            # neither prefix nor exponent
            exponent = 0
        
        # raise quantity to its exponent
        number = base * Decimal(10) ** Decimal(exponent)

        # return float equivalent
        return float(number), results.group(5)

    @classmethod
    def unit_exponent(cls, prefix):
        """Return exponent equivalent of unit prefix

        :param prefix: unit prefix
        :type prefix: str
        :return: exponent
        :rtype: int
        :raises ValueError: if prefix is unknown
        """

        if prefix in cls.PREFIX_ALIASES:
            # use real prefix for this alias
            prefix = cls.PREFIX_ALIASES[prefix]

        # find exponent in prefix dict
        for exponent, this_prefix in cls.UNIT_PREFICES.items():
            if this_prefix == prefix:
                return exponent

        # prefix not found
        raise ValueError("Unknown prefix")
