import logging
import locale

PROGRAM = "zero"
DESCRIPTION = "Linear circuit simulator"

# get version
try:
    from ._version import version as __version__
except ImportError:
    # packaging resources are not installed
    __version__ = '?.?.?'

try:
    from matplotlib import rcParams
    from .config import ZeroConfig

    # get config
    CONF = ZeroConfig()

    # update Matplotlib options with overrides from config
    rcParams.update(CONF["plot"]["matplotlib"])
except ImportError:
    # matplotlib and/or numpy not installed
    pass

# Make Circuit class available from main package.
# This is placed here because dependent imports need the code above.
from .circuit import Circuit

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

def add_log_handler(logger, handler=None, format_str="%(name)-25s - %(levelname)-8s - %(message)s"):
    if handler is None:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(handler)

# create base logger
LOGGER = logging.getLogger(__name__)
add_log_handler(LOGGER)

def set_log_verbosity(level, logger=None):
    """Enable logging to stdout with a certain level"""
    if logger is None:
        logger = LOGGER

    logger.setLevel(level)
