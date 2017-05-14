import logging
import itertools
from resistor import Resistor

class Regulator(object):
    TYPE_LM317 = 1

    def __init__(self, reg_type):
        self.type = int(reg_type)

    def resistors_for_voltage(self, voltage, n_values, series=None, *args, **kwargs):
        voltage = float(voltage)
        n_values = int(n_values)
        resistor = Resistor(series)

        # get resistor values
        values = resistor.get_values(*args, **kwargs)

        # get possible resistor pairs
        logging.getLogger("regulator").debug("Calculating resistor combinations")
        combinations = list(itertools.combinations(values, 2))

        # calculate voltages
        logging.getLogger("regulator").debug("Calculating regulator voltages")
        voltages = list(map(self.regulated_voltage, combinations))
        logging.getLogger("regulator").debug("Found %d combinations", len(voltages))

        # add keys
        voltages_with_keys = [(i, voltages[i]) for i in range(len(voltages))]

        # sorted absolute voltage differences
        logging.getLogger("regulator").debug("Finding closest voltage matches")
        sorted_abs = sorted(voltages_with_keys, key=lambda i: abs(i[1] - voltage))

        # return the lowest n_values voltages, and corresponding resistors
        return [(voltages[i], *combinations[i]) for i, _ in sorted_abs[:n_values]]

    def regulated_voltage(self, resistors):
        if self.type is self.TYPE_LM317:
            return 1.25 * (1 + float(resistors[1]) / float(resistors[0]))
        else:
            raise ValueError("Unknown regulator type")
