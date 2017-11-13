import logging
import locale

# suppress warnings when the user code does not include a handler
logging.getLogger().addHandler(logging.NullHandler())

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

__version__ = "0.1.0"
DESCRIPTION = "Electronics calculator and simulator utility"
PROGRAM = "electronics"
