.. currentmodule:: zero.components

Passive components
------------------

Passive components do not produce or amplify signals, but only apply an impedance to their input.
They have two nodes, :attr:`~.PassiveComponent.node1` and :attr:`~.PassiveComponent.node2`. The node
order does not matter. Passive components have a complex, frequency dependent
:meth:`~.PassiveComponent.impedance`; the specific component type - resistor, capacitor or inductor
- governs how this impedance behaves as a function of frequency.

Resistors
=========

.. code-block:: python

   >>> from zero.components import Resistor

Resistors have a real impedance, i.e. a resistance, with no frequency dependence. This resistance
has units of ohm (Ω). A resistor object can be instantiated by providing the resistance and the name
of two nodes:

.. code-block:: python

   >>> r = Resistor(value="430k", node1="n1", node2="n2")

The resistance can be changed using the resistor's :meth:`~Resistor.resistance` property:

.. code-block:: python

   >>> r.resistance = "1.1M"

In a circuit, resistor produce :class:`Johnson noise <.ResistorJohnsonNoise>`.

Capacitors
==========

.. code-block:: python

   >>> from zero.components import Capacitor

Capacitors have an imaginary, frequency dependent impedance determined by its capacitance in units
of farad (F). A capacitor object can be instantiated by providing the capacitance and the name of
two nodes:

.. code-block:: python

   >>> c = Capacitor(value="47n", node1="n1", node2="n2")

The capacitance can be changed using the capacitor's :meth:`~Capacitor.capacitance` property:

.. code-block:: python

   >>> c.capacitance = "100n"

Capacitors are considered ideal and do not produce noise.

Inductors
=========

.. code-block:: python

   >>> from zero.components import Inductor

Inductors have an imaginary, frequency dependent impedance determined by its inductance in units of
henry (H). An inductor object can be instantiated by providing the inductance and the name of two
nodes:

.. code-block:: python

   >>> l = Inductor(value="1.6u", node1="n1", node2="n2")

The inductance can be changed using the inductor's :meth:`~Inductor.inductance` property:

.. code-block:: python

   >>> l.inductance = "2.2u"

Inductors are considered ideal and do not produce noise.

A pair of inductors can also be configured as mutual inductors, allowing for transformers to be
simulated.
