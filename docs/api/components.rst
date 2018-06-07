.. currentmodule:: circuit.components

#########################
Components in the circuit
#########################

.. code-block:: python

   >>> from circuit.components import Resistor, Capacitor, Inductor, OpAmp

======================
What is a 'component'?
======================

A :class:`component <Component>` represents a circuit device which sources or
sinks current, and produces voltage drops between its :class:`nodes <Node>`.
:class:`Passive <PassiveComponent>` components such as :class:`resistors <Resistor>`,
:class:`capacitors <Capacitor>` and :class:`inductors <Inductor>` do not produce or
amplify signals, but only apply an impedance to their input. Active components such as
:class:`op-amps <OpAmp>` can source current.

Instantiated components may be added to :class:`circuits <.Circuit>` using
:meth:`.add_component`; however, the methods
:meth:`.add_resistor`, :meth:`.add_capacitor`,
:meth:`.add_inductor` and :meth:`.add_opamp` allow
components to be created and added to a circuit at the same time.

-----------------------
Component noise sources
-----------------------

Some components directly produce :class:`noise at a node they are connected to <NodeNoise>`.
Others create :class:`noise affecting current flow <ComponentNoise>`. The type and amount
of noise depends on the component; for example, :class:`capacitors <Capacitor>` do not
produce noise, whereas :class:`resistors <Resistor>` do (:class:`Johnson noise <JohnsonNoise>`).

---------------------------
Setting a component's value
---------------------------

A :class:`passive component <PassiveComponent>`'s :attr:`~PassiveComponent.value`
may be altered. The type may be :class:`int` or :class:`float` to directly
specify the numerical value, or alternatively an SI formatted string may be
provided, e.g.:

* 1.1k
* 2.2nF
* 1e-9 Hz
* 6.4 kHz

The provided string will be parsed by :class:`.SIFormatter` into an
appropriate :class:`float`. The unit, if provided, will be ignored.

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

-------
Details
-------

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