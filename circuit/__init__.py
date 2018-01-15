import logging
import locale
from matplotlib import rcParams

from circuit.config import CircuitConfig

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

__version__ = "0.2.7"
DESCRIPTION = "Linear circuit simulator based on Gerhard Heinzel's LISO"
PROGRAM = "circuit"

# get config
CONF = CircuitConfig()

# update Matplotlib options with overrides from config
rcParams.update(CONF["matplotlib"])

def logging_on(level=logging.DEBUG,
               format_str="%(name)-8s - %(levelname)-8s - %(message)s"):
    # enable logging to stdout
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_str))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
