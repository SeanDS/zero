"""Base circuit elements"""

import abc


class ElementNotFoundError(Exception):
    def __init__(self, name, message="element '%s' not found", *args, **kwargs):
        # apply name to message
        message = message % name

        # call parent constructor
        super().__init__(message, *args, **kwargs)

        self.name = name


class BaseElement(metaclass=abc.ABCMeta):
    """Represents a source or sink of a function, with a unit.

    This is an abstract representation of components, nodes or noise sources.
    """
    # Element type. Represents whether this element behaves like a component, node, etc.
    ELEMENT_TYPE = None
    # Unit used for admittance calculations.
    ELEMENT_UNIT = None

    @property
    def element_type(self):
        return self.ELEMENT_TYPE

    @property
    def element_unit(self):
        return self.ELEMENT_UNIT


class GenericElement(BaseElement):
    """Represents a generic element with custom unit.

    This is used in place of components and nodes when creating functions with non-circuit elements.
    """
    ELEMENT_TYPE = "__custom__"

    def __init__(self, name, unit):
        self.name = str(name)
        self._unit = str(unit)

    @property
    def element_unit(self):
        return self._unit

    @property
    def label(self):
        return self.name
