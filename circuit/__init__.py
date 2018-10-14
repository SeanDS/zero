import logging
import locale
from pkg_resources import get_distribution, DistributionNotFound

PROGRAM = "circuit"
DESCRIPTION = "Linear circuit simulator"

# get version
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # packaging resources are not installed
    __version__ = '?'

try:
    from matplotlib import rcParams
    from .config import CircuitConfig

    # get config
    CONF = CircuitConfig()

    # update Matplotlib options with overrides from config
    rcParams.update(CONF["matplotlib"])
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

# create logger
LOG_FORMAT_STR = "%(name)-25s - %(levelname)-8s - %(message)s"
LOG_HANDLER = logging.StreamHandler()
LOG_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT_STR))
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(LOG_HANDLER)

def set_log_verbosity(level):
    """Enable logging to stdout with a certain level"""
    LOGGER.setLevel(level)
