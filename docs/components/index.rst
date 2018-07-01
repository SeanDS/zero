Components
==========

.. code-block:: python

   >>> from circuit.components import Resistor, Capacitor, Inductor, OpAmp

======================
What is a 'component'?
======================

A :class:`component <.Component>` represents a circuit device which sources or
sinks current, and produces voltage drops between its :class:`nodes <.Node>`.
:class:`Passive <.PassiveComponent>` components such as :class:`resistors <.Resistor>`,
:class:`capacitors <.Capacitor>` and :class:`inductors <.Inductor>` do not produce or
amplify signals, but only apply an impedance to their input. Active components such as
:class:`op-amps <.OpAmp>` can source current.

Instantiated components may be added to :class:`circuits <.Circuit>` using
:meth:`.add_component`; however, the methods :meth:`.add_resistor`, :meth:`.add_capacitor`,
:meth:`.add_inductor` and :meth:`.add_opamp` allow components to be created and added to
a circuit at the same time.

-----------------------
Component noise sources
-----------------------

Some components directly produce noise at a node they are connected to (:class:`.NodeNoise`).
Others create noise affecting current flow (:class:`.ComponentNoise`). The type and amount
of noise depends on the component; for example, :class:`capacitors <.Capacitor>` do not
produce noise, whereas :class:`resistors <.Resistor>` do (:class:`Johnson noise <.JohnsonNoise>`).

---------------------------
Setting a component's value
---------------------------

A passive component's :attr:`~.PassiveComponent.value`
may be altered. The type may be :class:`int` or :class:`float` to directly
specify the numerical value, or alternatively an SI formatted string may be
provided, e.g.:

* :code:`1.1k`
* :code:`2.2nF`
* :code:`1e-9 Hz`
* :code:`6.4 kHz`

The provided string will be parsed by :class:`.Quantity` into an appropriate :class:`float`
representation.

.. toctree::
    :maxdepth: 2

    passive-components
    op-amps
