import locale

__version__ = "0.1.0"

PROG = "electronics"
DESC = "Electronics calculator and simulator utility"

# use default locale (required for number formatting in log warnings)
locale.setlocale(locale.LC_ALL, "")
