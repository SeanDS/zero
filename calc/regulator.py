import logging
import itertools
import heapq

from .resistor import Set
from .misc import _n_perm_k, _print_progress

class Regulator(object):
    TYPE_LM317 = 1
    TYPE_LM337 = 2

    def __init__(self, reg_type):
        self.type = reg_type

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, reg_type):
        reg_type = str(reg_type).lower()

        if reg_type == "lm317":
            self._type = self.TYPE_LM317
            logging.getLogger("regulator").info("Using LM317 regulator")
        elif reg_type == "lm337":
            self._type = self.TYPE_LM337
            logging.getLogger("regulator").info("Using LM337 regulator")
        else:
            raise ValueError("Unknown regulator type")

    def resistors_for_voltage(self, voltage, resistor_set, n_values=3,
                              progress=True):
        voltage = float(voltage)
        n_values = int(n_values)

        # calculate number of results
        n_permutations = _n_perm_k(resistor_set.n_combinations(), 2)
        logging.getLogger("regulator").info("Calculating %i permutations",
                                            n_permutations)

        # get regulator resistor pairs using resistor set combinations
        permutations = itertools.permutations(resistor_set.combinations(), 2)

        # calculate voltages for resistor pairs
        logging.getLogger("regulator").debug("Generating regulator voltages")
        voltages = self.regulated_voltages(permutations)

        if progress:
            # add progress bar between voltage and heapq generators
            voltages = _print_progress(voltages, n_permutations)

        # sorted absolute voltage differences
        logging.getLogger("regulator").debug("Finding closest voltage matches")
        return heapq.nsmallest(n_values, voltages, key=lambda i: abs(i[0] - voltage))

    def regulated_voltages(self, resistor_pairs):
        for pair in resistor_pairs:
            yield (self._regulated_voltage(pair), *pair)

    def _regulated_voltage(self, resistors):
        if self.type in [self.TYPE_LM317, self.TYPE_LM337]:
            return 1.25 * (1 + float(resistors[1].resistance)
                               / float(resistors[0].resistance))
        else:
            raise ValueError("Unknown regulator type")
