import logging
import locale
from pkg_resources import get_distribution, DistributionNotFound

# get version
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

PROGRAM = "datasheet"
DESCRIPTION = "Datasheet grabber"

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

def logging_on(level=logging.DEBUG, format_str="%(name)-25s - %(levelname)-8s - %(message)s"):
    """Enable logging to stdout"""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_str))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(level)
