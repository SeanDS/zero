import locale
import version

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")

__version__ = version.VERSION
DESCRIPTION = version.DESCRIPTION
PROGRAM = version.PROGRAM
