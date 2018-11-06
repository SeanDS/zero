########
Circuits
########

.. code-block:: python

   >>> from zero import Circuit

====================
What is a 'circuit'?
====================

A :class:`circuit <.Circuit>` describes a collection of :class:`components <.Component>`
connected at :class:`nodes <.Node>`. It may contain :class:`resistors <.Resistor>`,
:class:`capacitors <.Capacitor>`, :class:`inductors <.Inductor>` and
:class:`op-amps <.OpAmp>`, and the circuit can be supplied with an :class:`input <.Input>`
in order to produce a current through and voltage across these components.
