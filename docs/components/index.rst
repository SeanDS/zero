.. include:: /defs.txt

.. currentmodule:: zero.components

Components
==========

.. code-block:: python

   >>> from zero.components import Resistor, Capacitor, Inductor, OpAmp

.. toctree::
    :maxdepth: 2

    passive-components
    op-amps
    noise

What is a 'component'?
----------------------

A :class:`component <.Component>` represents a circuit device which sources or sinks current, and
produces voltage drops between its :class:`nodes <.Node>`. :class:`Passive <.PassiveComponent>`
components such as :class:`resistors <.Resistor>`, :class:`capacitors <.Capacitor>` and
:class:`inductors <.Inductor>` do not produce or amplify signals, but only apply an impedance to
their input. Active components such as :class:`op-amps <.OpAmp>` can source current.

Instantiated components may be added to :ref:`circuits <circuit/index:Circuits>` using
:meth:`.add_component`; however, the methods :meth:`.add_resistor`, :meth:`.add_capacitor`,
:meth:`.add_inductor` and :meth:`.add_opamp` allow components to be created and added to a circuit
at the same time, and avoid the need to import them directly.

.. note::

    The recommended way to add components to a circuit is to use the :meth:`.add_resistor`,
    :meth:`.add_capacitor`, :meth:`.add_inductor` and :meth:`.add_opamp` methods provided by
    :class:`.Circuit`. These offer the same functionality as when creating
    component objects directly, but avoid the need to directly import the component classes into
    your script.

Component names
---------------

Components may be provided with a name on creation using the ``name`` keyword argument, i.e.

.. code-block:: python

    >>> r = Resistor(name="r1", value="430k", node1="n1", node2="n2")

or

.. code-block:: python

    >>> from zero import Circuit
    >>> circuit = Circuit()
    >>> circuit.add_resistor(name="rin", value="430k", node1="n1", node2="n2")

Names can also be set using the :attr:`~.Component.name` property:

.. code-block:: python

    >>> r.name = "r1"

Component names can be used to retrieve components from circuits:

.. code-block:: python

    >>> r = circuit["rin"]
    >>> print(r)
    rin [in=n1, out=n2, R=430.00k]

Component names must be unique within a given circuit. When trying to add a component to a circuit
where its name is already used by another circuit component, a :class:`ValueError` is raised.

.. note::

    Component names do not need to be unique within the global namespace. That means components with
    different values or nodes can have the same name as long as they are not part of the same
    circuit.

Naming of components is not required; however, when a component is added to a circuit it is assigned
a name if it does not yet have one. This name uses a prefix followed by a number (the lowest
positive integer not resulting in a name which matches that of a component already present in the
circuit). The character(s) used depend on the component type:

=========  ======  =======
Component  Prefix  Example
=========  ======  =======
Resistor   r       r1
Capacitor  c       c1
Inductor   l       l1
Op-amp     op      op1
=========  ======  =======

Setting a component's value
---------------------------

A passive component's :attr:`~.PassiveComponent.value` may be altered. First, get the component:

.. code:: python

    c1 = circuit["c1"]

You can then set the value using the object's :attr:`~.PassiveComponent.value` attribute:

.. code:: python

    c1.value = "1u"

In the above example, the string is parsed parsed by :class:`.Quantity` into an appropriate
:class:`float` representation. You may also specify a :class:`float` or :class:`int` directly:

.. code:: python

    c1.value = 1e-6

You may also provide a string with units or scales:

.. code:: python

    # Quantity with scale factor and unit.
    c1.value = "2.2nF"

The above value is parsed as ``2.2e-9``, with unit ``F``. The unit is stored alongside the numeric
part within the object, and the unit will be printed alongside the component's value when it is
displayed.

.. note::
    Units are just for display and are not used for any calculations. Be careful when specifying
    units which differ from those used internally by |Zero|.
