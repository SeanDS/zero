"""Formatting functionality for numbers with units"""

import abc
import math
import re
from decimal import Decimal

class BaseFormatter(metaclass=abc.ABCMeta):
    """Abstract class for all formatters"""

    @abc.abstractclassmethod
    def format(cls, value, unit=""):
        """Format the specified value and unit for display.

        Parameters
        ----------
        value : :class:`float`
            The value to format.
        unit : :class:`str`, optional
            The unit to append to the value.
        
        Returns
        -------
        :class:`str`
            Formatted value and unit
        """
        raise NotImplementedError

    @staticmethod
    def exponent(value):
        """Calculate the exponent of 10 corresponding to the specified value.

        Parameters
        ----------
        value : :class:`float`
            The value.
        
        Returns
        -------
        :class:`float`
            The exponent of 10 for the specified value.
        """
        if value == 0:
            return 0

        # calculate log of value to get exponent
        return math.log(abs(value), 10)

class SIFormatter(BaseFormatter):
    """SI unit formatter.

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
        """Format the specified value and unit for display.

        Parameters
        ----------
        value : :class:`float`
            The value to format.
        unit : :class:`str`, optional
            The unit to append to the value.
        
        Returns
        -------
        :class:`str`
            Formatted value and unit
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
        """Get unit prefices, including aliases.

        Yields
        ------
        :class:`str`
            Unit prefices/aliases
        """
        yield from cls.UNIT_PREFICES.values()
        yield from cls.PREFIX_ALIASES.keys()

    @classmethod
    def parse(cls, quantity):
        """Parse quantity as a number.


        Parameters
        ----------
        quantity : :class:`str`
            Value string to parse as a number.
        
        Returns
        -------
        number : :class:`float`
            The numeric representation of the quantity.
        unit : :class:`str`
            The parsed unit. If no unit is found, `None` is returned.
        """
        # don't need to handle units if there aren't any
        if isinstance(quantity, (int, float)):
            return float(quantity), None

        # find floating point numbers and optional unit prefix in string
        results = re.match(cls.VALUE_REGEX, quantity)

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
        """Find exponent equivalent of unit prefix.

        Parameters
        ----------
        prefix : :class:`str`
            The unit prefix.
        
        Returns
        -------
        :class:`int`
            The quantity's exponent.
        
        Raises
        ------
        ValueError
            If the specified prefix is unknown.
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

class Quantity(float):
    """Container for numeric values and their associated units
    
    Parameters
    ----------
    value : :class:`float`, :class:`str`, :class:`Quantity`
        The quantity. SI units and prefices can be specified and are recognised.
    unit : :class:`str`, optional
        The quantity's unit. This can be used to directly specify the unit associated with
        the specified `value`.
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

    def __new__(cls, value, unit=None):
        if isinstance(value, Quantity):
            number = float(value)

            if value.unit:
                unit = value.unit
        elif isinstance(value, str):
            number, unit = cls.parse(value)
        else:
            # assume float
            number = value
    
        # create float from identified number
        self = float.__new__(cls, number)
        self.unit = unit

        return self

    @classmethod
    def parse(cls, quantity):
        """Parse quantity as a number.

        Parameters
        ----------
        quantity : :class:`str`
            Value string to parse as a number.
        
        Returns
        -------
        number : :class:`float`
            The numeric representation of the quantity.
        unit : :class:`str`
            The parsed unit. If no unit is found, `None` is returned.
        """
        # don't need to handle units if there aren't any
        if isinstance(quantity, (int, float)):
            return float(quantity), None

        # find floating point numbers and optional unit prefix in string
        results = re.match(cls.VALUE_REGEX, quantity)

        # unit is fifth match
        unit = results.group(5)

        # first result should be the base number
        base = Decimal(results.group(1))

        # special case: parse "1.23E" as 1.23e15
        if results.group(2) == "E" and not results.group(3):
            exponent = 15
        else:
            # handle exponent
            if results.group(3) or results.group(4):
                exponent = 0

                if results.group(3):
                    # exponent specified directly
                    exponent += Decimal(results.group(3))
                
                if results.group(4):
                    # exponent specified as unit prefix
                    exponent += cls.unit_exponent(results.group(4))
            else:
                # neither prefix nor exponent
                exponent = 0
        
        # raise quantity to its exponent
        number = base * Decimal(10) ** Decimal(exponent)

        # return float equivalent
        return float(number), unit

    @classmethod
    def unit_exponent(cls, prefix):
        """Find exponent equivalent of unit prefix.

        Parameters
        ----------
        prefix : :class:`str`
            The unit prefix.
        
        Returns
        -------
        :class:`int`
            The quantity's exponent.
        
        Raises
        ------
        ValueError
            If the specified prefix is unknown.
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

    def format(self, unit=True, si=True, precision=None):
        """Format the specified value and unit for display.

        Parameters
        ----------
        unit : :class:`bool`, optional
            Whether to display the quantity's unit.
        si : :class:`bool`, optional
            Whether to show quantity with scale factors using SI notation, e.g. "k" for 10e3.
            If `False`, the value defaults to using standard float notation.
        precision : :class:`int`, optional
            Number of decimal places to display quantity with. Defaults to full precision.
        
        Returns
        -------
        :class:`str`
            Formatted quantity.
        """
        value = float(self)

        if si:
            # multiple of 3 corresponding to the unit prefix list
            exp = int(self.exponent(value) // 3) * 3

            # check for out of range exponents
            if exp < min(self.UNIT_PREFICES):
                exp = min(self.UNIT_PREFICES)
            elif exp > max(self.UNIT_PREFICES):
                exp = max(self.UNIT_PREFICES)

            unit_prefix = self.UNIT_PREFICES[exp]

            # scale value to account for scale
            value *= 10 ** -exp
        else:
            unit_prefix = ""

        if precision is not None:
            value = "%.*f" % (int(precision), value)

        suffix = unit_prefix
        
        if unit and self.unit:
            suffix += self.unit

        return "{0} {1}".format(value, suffix) if suffix else str(value)

    def exponent(self, value):
        """Calculate the exponent of 10 corresponding to the specified value.

        Parameters
        ----------
        value : :class:`float`
            The value.
        
        Returns
        -------
        :class:`float`
            The exponent of 10 for the specified value.
        
        Raises
        ------
        ValueError
            If `value` is negative.
        """
        if value == 0:
            return 0

        # calculate log of value to get exponent
        return math.log(value, 10)

    def __str__(self):
        # quantity value as string
        string = str(self.real)

        if self.unit is not None:
            string = "{value} {unit}".format(value=string, unit=self.unit)
        
        return string
    
    def __repr__(self):
        return str(self)