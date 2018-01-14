.. currentmodule:: circuit.components

#######################
The :class:`Components`
#######################

.. code-block:: python

   >>> from circuit.components import Resistor, Capacitor, Inductor, OpAmp, Input

======================
What is a 'component'?
======================

A :class:`component <Component>` represents a circuit device which sources or
sinks current, and produces voltage drops between its :class:`nodes <Node>`.
:class:`Passive <PassiveComponent>` components such as
:class:`resistors <Resistor>`, :class:`capacitors <Capacitor>` and
:class:`inductors <Inductor>` do not produce or amplify signals, but only apply
an impedance to their input. Active components such as :class:`op-amps <OpAmp>`
can source current.

===============
Class reference
===============

This reference contains the following `class` entries:

.. autosummary::
   :nosignatures:

   Component
   PassiveComponent
   Resistor
   Capacitor
   Inductor
   OpAmp
   Input
   Node
   Noise
   ComponentNoise
   NodeNoise
   VoltageNoise
   JohnsonNoise
   CurrentNoise
   BaseEquation
   ComponentEquation
   NodeEquation
   BaseCoefficient
   ImpedanceCoefficient
   CurrentCoefficient
   VoltageCoefficient

.. autoclass:: Component
    :members:

.. autoclass:: PassiveComponent
    :members:

.. autoclass:: Resistor
    :members:

.. autoclass:: Capacitor
    :members:

.. autoclass:: Inductor
    :members:

.. autoclass:: OpAmp
    :members:

.. autoclass:: Input
    :members:

.. autoclass:: Node
    :members:

.. autoclass:: Noise
    :members:

.. autoclass:: ComponentNoise
    :members:

.. autoclass:: NodeNoise
    :members:

.. autoclass:: VoltageNoise
    :members:

.. autoclass:: JohnsonNoise
    :members:

.. autoclass:: CurrentNoise
    :members:

.. autoclass:: BaseEquation
    :members:

.. autoclass:: ComponentEquation
    :members:

.. autoclass:: NodeEquation
    :members:

.. autoclass:: BaseCoefficient
    :members:

.. autoclass:: ImpedanceCoefficient
    :members:

.. autoclass:: CurrentCoefficient
    :members:

.. autoclass:: VoltageCoefficient
    :members:
