"""LISO parser utilities"""

from ..components import Component, Node


def liso_sort_key(function):
    """Sort items by their sink type, then source and sink names."""
    order = [Node.SOURCE_SINK_UNIT, Component.SOURCE_SINK_UNIT]
    return order.index(function.sink.SOURCE_SINK_UNIT), str(function.source), str(function.sink)
