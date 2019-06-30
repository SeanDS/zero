.. currentmodule:: zero.analysis.ac.signal

Small AC signal analysis
========================

The small AC signal analysis calculates the signal at all :class:`nodes <.Node>` and
:class:`components <.Component>` within a circuit due to either a voltage or a current applied to
the circuit's input. The input is unity, meaning that the resulting signals represent the
:ref:`responses <data/index:Responses>` from the input to each node or component. The analysis
assumes that the input is small enough not to influence the operating point and gain of the circuit.
