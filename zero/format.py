"""Formatting functionality for numbers with units"""

import re
import logging

LOGGER = logging.getLogger(__name__)


class Quantity(float):
    """Container for numeric values and their associated units.

    Partially based on `QuantiPhy <https://github.com/KenKundert/quantiphy>`_.

    Parameters
    ----------
    value : :class:`float`, :class:`str`, :class:`Quantity`
        The quantity. SI units and prefices can be specified and are recognised.
    unit : :class:`str`, optional
        The quantity's unit. This can be used to directly specify the unit associated with
        the specified `value`.
    """
    # default display precision
    DEFAULT_PRECISION = 4

    # input scale mappings
    MAPPINGS = {
        'Y': 24,
        'Z': 21,
        'E': 18,
        'P': 15,
        'T': 12,
        'G': 9,
        'M': 6,
        'k': 3,
        'c': -2, # only available for input, not used in output
        'm': -3,
        'u': -6,
        'µ': -6,
        'n': -9,
        'p': -12,
        'f': -15,
        'a': -18,
        'z': -21,
        'y': -24
    }

    # scale factors every 3rd decade, in order from 0
    LARGE_SCALES = "kMGTPEZY"
    SMALL_SCALES = "mµnpfazy"

    # output scale factors (only display these scales regardless of input)
    OUTPUT_SCALES = "TGMkmµnpf"

    # regular expression to find values with unit prefixes and units in text
    VALUE_REGEX_STR = (r"^([+-]?\d*\.?\d*)" # base
                       r"([eE]([+-]?\d*\.?\d*))?\s*" # numeric exponent
                       r"([yzafpnuµmkMGTPEZY])?" # unit prefix
                       r"(s|A|rad|Hz|W|C|V|F|Ω|H|°C)?") # SI unit
    VALUE_REGEX = re.compile(VALUE_REGEX_STR)

    def __new__(cls, value, unit=None):
        if isinstance(value, Quantity):
            number = float(value)
            mantissa = value._mantissa
            scale = value._scale

            if value.unit:
                unit = value.unit
        elif isinstance(value, str):
            number, mantissa, scale, parsed_unit = cls.parse(value, unit)

            if unit is not None and parsed_unit is not None and unit != parsed_unit:
                LOGGER.warning("overriding detected unit '%s' with specified unit '%s'",
                               parsed_unit, unit)
            else:
                unit = parsed_unit
        else:
            # assume float
            number = value
            mantissa = None
            scale = None

        # create object from identified information
        self = float.__new__(cls, number)
        self._mantissa = mantissa
        self._scale = scale
        self.unit = unit

        return self

    @classmethod
    def parse(cls, quantity, unit=None):
        """Parse quantity as a number.

        Parameters
        ----------
        quantity : :class:`str`
            Value string to parse as a number.
        unit : :class:`str`, optional
            The quantity's unit.

        Returns
        -------
        value : :class:`float`
            The numeric representation of the quantity.
        mantissa : :class:`str`
            The quantity's mantissa. Provided to help avoid floating point precision
            display errors.
        scale : :class:`float`
            The quantity's scale factor.
        unit : :class:`str`
            The parsed unit. If no unit is found, `None` is returned.
        """
        # don't need to handle units if there aren't any
        if isinstance(quantity, (int, float)):
            return float(quantity), None

        # find floating point numbers and optional unit prefix in string
        results = re.match(cls.VALUE_REGEX, quantity)

        # mantissa is first match
        mantissa = results.group(1)
        # scale is the fourth match
        scale = results.group(4)
        # unit is fifth match
        unit = results.group(5)

        if not mantissa:
            raise ValueError(f"unrecognised quantity '{quantity}'")

        # convert value to float
        value = float(mantissa)

        # special case: parse "1.23E" as 1.23e18
        if results.group(2) == "E" and results.group(3) == "":
            exponent = 18
            scale = "E"
        else:
            # handle exponent
            if results.group(3) or results.group(4):
                exponent = 0

                if results.group(3):
                    # exponent specified directly
                    exponent += float(results.group(3))

                if results.group(4):
                    # exponent specified as unit prefix
                    exponent += cls.MAPPINGS[results.group(4)]
            else:
                # neither prefix nor exponent
                exponent = 0

        # raise value to the intended exponent
        value *= 10 ** exponent

        return value, mantissa, scale, unit

    def format(self, show_unit=True, show_si=True, precision=None):
        """Format the specified value and unit for display.

        Parameters
        ----------
        show_unit : :class:`bool`, optional
            Whether to display the quantity's unit.
        show_si : :class:`bool`, optional
            Whether to show quantity with scale factors using SI notation, e.g. "k" for 10e3.
            If `False`, the value defaults to using standard float notation.
        precision : :class:`int` or 'full', optional
            Number of decimal places to display quantity with. If "full", uses the precision
            of the number as originally specified. When "full" is used, but `show_unit` and
            `show_si` are both `False`, then the decimal point may be moved.

        Returns
        -------
        :class:`str`
            Formatted quantity.
        """
        if precision is None:
            precision = self.DEFAULT_PRECISION

        if precision == "full" and self._mantissa is not None:
            # parsed mantissa and scale factor
            mantissa = self._mantissa
            scale = self._scale

            # convert scale factor to integer exponent
            try:
                exp = int(scale)
            except ValueError:
                if scale:
                    exp = int(self.MAPPINGS[scale])
                else:
                    exp = 0

            # add decimal point to mantissa if missing
            mantissa += '' if '.' in mantissa else '.'
            # strip off leading zeros and break into components
            whole, frac = mantissa.strip('0').split('.')

            if whole == "":
                # remove leading zeros from fractional part
                orig_len = len(frac)
                frac = frac.lstrip('0')

                if frac:
                    whole = frac[:1]
                    frac = frac[1:]
                    exp -= orig_len - len(frac)
                else:
                    # stripping off zeros left us with nothing, this must be 0
                    whole = '0'
                    frac = ''
                    exp = 0

            # reconstruct the mantissa
            mantissa = whole[0] + '.' + whole[1:] + frac
            exp += len(whole) - 1
        else:
            if precision == "full":
                # no parsed mantissa available; use default precision
                precision = self.DEFAULT_PRECISION

            # Get float value.
            number = self.real

            # Split number into components.
            number = "%.*e" % (precision, number)
            mantissa, exp = number.split("e")
            exp = int(exp)

        # scale factor
        index = exp // 3
        shift = exp % 3
        scale = "e%d" % (exp - shift)

        if index == 0:
            scale = ''
        elif show_si:
            if index > 0:
                if index <= len(self.LARGE_SCALES):
                    if self.LARGE_SCALES[index-1] in self.OUTPUT_SCALES:
                        scale = self.LARGE_SCALES[index-1]
            else:
                index = -index

                if index <= len(self.SMALL_SCALES):
                    if self.SMALL_SCALES[index-1] in self.OUTPUT_SCALES:
                        scale = self.SMALL_SCALES[index-1]

        # shift the decimal place as needed
        sign = '-' if mantissa[0] == '-' else ''
        mantissa = mantissa.lstrip('-').replace('.', '')
        mantissa += (shift + 1 - len(mantissa)) * '0'
        mantissa = sign + mantissa[0:(shift+1)] + '.' + mantissa[(shift+1):]

        # get rid of trailing decimal points and leading + if present
        mantissa = mantissa.rstrip('.')
        mantissa = mantissa.lstrip('+')

        if show_unit and self.unit is not None:
            if scale in self.MAPPINGS:
                # standard suffix
                fmt_str = "{mantissa} {scale}{unit}"
            else:
                # scientific notation
                if self.unit is not None:
                    fmt_str = "{mantissa}{scale} {unit}"
                else:
                    fmt_str = "{mantissa}{scale}"
        else:
            fmt_str = "{mantissa}{scale}"

        return fmt_str.format(mantissa=mantissa, scale=scale, unit=self.unit)

    def __str__(self):
        return self.format()
