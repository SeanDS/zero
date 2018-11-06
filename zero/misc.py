"""Miscellaneous functions"""

import abc
import numpy as np


class Singleton(abc.ABCMeta):
    """Metaclass implementing the singleton pattern

    This ensures that there is only ever one instance of a class that
    inherits this one.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by class
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class NamedInstance(abc.ABCMeta):
    """Metaclass to implement a single named instance pattern

    This ensures that there is only ever one instance of a class with a specific
    name, as provided by the "name" constructor argument.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by name
    _names = {}

    def __call__(cls, name, *args, **kwargs):
        name = name.lower()

        if name not in cls._names:
            cls._names[name] = super().__call__(name, *args, **kwargs)

        return cls._names[name]


def db(magnitude):
    """Calculate (power) magnitude in decibels

    :param magnitude: magnitude
    :type magnitude: Numeric or :class:`np.array`
    :return: dB magnitude
    :rtype: Numeric or :class:`np.array`
    """

    return 20 * np.log10(magnitude)
