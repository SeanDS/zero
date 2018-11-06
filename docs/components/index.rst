.. include:: /defs.txt

Components
==========

.. code-block:: python

   >>> from zero.components import Resistor, Capacitor, Inductor, OpAmp

.. toctree::
    :maxdepth: 2

    passive-components
    op-amps

What is a 'component'?
----------------------

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

Component noise sources
-----------------------

Some components directly produce noise at a node they are connected to (:class:`.NodeNoise`).
Others create noise affecting current flow (:class:`.ComponentNoise`). The type and amount
of noise depends on the component; for example, :class:`capacitors <.Capacitor>` do not
produce noise, whereas :class:`resistors <.Resistor>` do (:class:`Johnson noise <.JohnsonNoise>`).

Setting a component's value
---------------------------

A passive component's :attr:`~.PassiveComponent.value` may be altered. First, get the component:

.. code:: python

    c1 = circuit["c1"]

You can then set the value using the object's :attr:`~.PassiveComponent.value` attribute:

.. code:: python

    # string
    c1.value = "1u"

In the above example, the string is parsed parsed by :class:`.Quantity` into an appropriate
:class:`float` representation. You may also specify a :class:`float` or :class:`int` directly:

.. code:: python

    # float
    c1.value = 1e-6

You may also provide a string with units or scales:

.. code:: python

    # string with scale factor and unit
    c1.value = "2.2nF"

The above value is parsed as ``2.2e-9``, with unit ``F``. The unit is stored alongside the numeric
part within the object, and the unit will be printed alongside the component's value when it is
displayed.

.. note::
    Units are just for display and are not used for any calculations. Be careful when specifying
    units which differ from those used internally by |Zero|.
