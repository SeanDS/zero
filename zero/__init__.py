import logging

PROGRAM = "zero"
DESCRIPTION = "Linear circuit simulator"

# Get package version.
try:
    from ._version import version as __version__
except ImportError:
    # Packaging resources are not installed.
    __version__ = '?.?.?'

try:
    from matplotlib import rcParams
    from .config import ZeroConfig
    # Get config.
    CONF = ZeroConfig()
    # Update Matplotlib options with overrides from config.
    rcParams.update(CONF["plot"]["matplotlib"])
except ImportError:
    # Matplotlib and/or numpy not installed.
    pass

# Make Circuit class available from main package.
# This is placed here because dependent imports need the code above.
from .circuit import Circuit

# Suppress warnings when the user code does not include a handler.
logging.getLogger().addHandler(logging.NullHandler())

def add_log_handler(logger, handler=None, format_str="{levelname}: {message} ({name})"):
    if handler is None:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_str, style="{"))
    logger.addHandler(handler)

# Create base logger.
LOGGER = logging.getLogger(__name__)
add_log_handler(LOGGER)

def set_log_verbosity(level, logger=None):
    """Enable logging to stdout with a certain level"""
    if logger is None:
        logger = LOGGER
    logger.setLevel(level)
