.. currentmodule:: zero.components

Passive components
------------------

Resistors
=========

Resistors have a real impedance, i.e. a resistance, with units of ohm (Ω). A resistor object can be
instantiated by providing the resistance and the name of two nodes:

.. code-block:: python

   >>> from zero.components import Resistor
   r = Resistor(value="430k", node1="n1", node2="n2")

The resistance can be changed using the resistor's :meth:`~Resistor.resistance` property:

.. code-block:: python

   r.resistance = "1.1M"

In a circuit, resistor produce :class:`Johnson noise <.JohnsonNoise>`.

Capacitors
==========

Capacitors have an imaginary impedance determined by its capacitance in units of farad (F). A
capacitor object can be instantiated by providing the capacitance and the name of two nodes:

.. code-block:: python

   >>> from zero.components import Capacitor
   c = Capacitor(value="47n", node1="n1", node2="n2")

The capacitance can be changed using the capacitor's :meth:`~Capacitor.capacitance` property:

.. code-block:: python

   c.capacitance = "100n"

Capacitors are considered ideal and do not produce noise.

Inductors
=========

Inductors have an imaginary impedance determined by its inductance in units of henry (H). An
inductor object can be instantiated by providing the inductance and the name of two nodes:

.. code-block:: python

   >>> from zero.components import Inductor
   l = Inductor(value="1.6u", node1="n1", node2="n2")

The inductance can be changed using the inductor's :meth:`~Inductor.inductance` property:

.. code-block:: python

   l.inductance = "2.2u"

Inductors are considered ideal and do not produce noise.

A pair of inductors can also be configured as mutual inductors, allowing for transformers to be
simulated.
