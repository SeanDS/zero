"""LISO parser utilities"""

from ..components import Component, Node


def liso_order_key(item):
    """Sort items by their sink type, the way LISO does it."""
    order = [Node.SOURCE_SINK_UNIT, Component.SOURCE_SINK_UNIT]
    return order.index(item.sink.SOURCE_SINK_UNIT)
