import logging
import locale

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

__version__ = "0.2.11"
# description must be one line
DESCRIPTION = "Linear circuit simulator based on Gerhard Heinzel's LISO"
PROGRAM = "circuit"

try:
    from matplotlib import rcParams
    from circuit.config import CircuitConfig

    # get config
    CONF = CircuitConfig()

    # update Matplotlib options with overrides from config
    rcParams.update(CONF["matplotlib"])
except ImportError:
    # matplotlib and/or numpy not installed
    pass

def logging_on(level=logging.DEBUG,
               format_str="%(name)-8s - %(levelname)-8s - %(message)s"):
    # enable logging to stdout
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_str))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
