import logging
import itertools
import heapq

from .resistor import Set

class Regulator(object):
    TYPE_LM317 = 1

    def __init__(self, reg_type):
        self.type = reg_type

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, reg_type):
        if str(reg_type).lower() == "lm317":
            self._type = self.TYPE_LM317
            logging.getLogger("regulator").info("Using LM317 regulator")
        else:
            raise ValueError("Unknwon regulator type")

    def resistors_for_voltage(self, voltage, n_values, series=None, *args, **kwargs):
        voltage = float(voltage)
        n_values = int(n_values)
        resistor_set = Set(series)

        # get resistor values
        values = resistor_set.combinations(*args, **kwargs)

        # get possible resistor pairs
        logging.getLogger("regulator").debug("Calculating resistor combinations")
        combinations = itertools.combinations(values, 2)

        # calculate voltages
        logging.getLogger("regulator").debug("Calculating regulator voltages")
        voltages = self.regulated_voltages(combinations)

        # sorted absolute voltage differences
        logging.getLogger("regulator").debug("Finding closest voltage matches")
        return heapq.nsmallest(n_values, voltages, key=lambda i: abs(i[0] - voltage))

    def regulated_voltages(self, resistor_pairs):
        for pair in resistor_pairs:
            yield (self._regulated_voltage(pair), *pair)

    def _regulated_voltage(self, resistors):
        if self.type is self.TYPE_LM317:
            return 1.25 * (1 + float(resistors[1].resistance)
                               / float(resistors[0].resistance))
        else:
            raise ValueError("Unknown regulator type")
