import logging
import locale

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

__version__ = "0.2.2"
DESCRIPTION = "Electronics calculator and simulator utility"
PROGRAM = "electronics"

def logging_on(level=logging.DEBUG,
               format_str="%(name)-8s - %(levelname)-8s - %(message)s"):
    # enable logging to stdout
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_str))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
