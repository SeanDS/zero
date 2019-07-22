"""User tools."""

from .elements import BaseElement, GenericElement
from .data import Series, Response


def create_response(source, sink, data, frequencies=None, source_unit=None, sink_unit=None):
    """Create a response between the specified source and sink.

    Parameters
    ----------
    source, sink : :class:`.BaseElement` or :class:`str`
        The response's source and sink. This can be either an existing component or a string. If
        the source or sink is a string, the corresponding `source_unit` or `sink_unit` must be
        specified.
    data : sequence or :class:`.Series`
        The response's series or complex magnitude.
    frequencies : sequence, optional
        The response's frequency vector. If `y1` is a :class:`.Series`, a ValueError is raised.
    source_unit, sink_unit : :class:`str`, optional
        The source and sink unit. This is required if the corresponding source or sink is not a
        circuit element.

    Returns
    -------
    :class:`.Response`
        The response.
    """
    if not isinstance(source, BaseElement):
        if source_unit is None:
            raise ValueError("source unit must be specified when the source is custom")
        source = GenericElement(source, source_unit)
    if not isinstance(sink, BaseElement):
        if sink_unit is None:
            raise ValueError("sink unit must be specified when the sink is custom")
        sink = GenericElement(sink, sink_unit)
    if not isinstance(data, Series):
        if frequencies is None:
            raise ValueError("frequencies must be specified when data is not a Series")
        data = Series(frequencies, data)
    return Response(source=source, sink=sink, series=data)
