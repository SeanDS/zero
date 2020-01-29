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

A circuit can be instantiated without arguments:

.. code-block:: python

   >>> circuit = Circuit()

You can print the circuit to retrieve a list of its constituents:

.. code-block:: python

   >>> print(circuit)
   Circuit with 0 components and 0 nodes

Circuits are only useful once you add components. This is achieved using the various ``add_``
methods, such as :meth:`.add_resistor`, :meth:`.add_capacitor`, :meth:`.add_inductor` and
:meth:`.add_opamp`.

====================
Circuit manipulation
====================

Circuits can be modified before and after applying :ref:`analyses <analyses/index:Analyses>`.
Circuit components can be removed with :meth:`.remove_component` or replaced with
:meth:`.replace_component`.

When a component is removed, any connected nodes shared by other components are preserved.

When a component is replaced with another one, its nodes are copied to the new component and the new
component's nodes are overwritten. The components being swapped must be compatible: the number of
nodes in the current and replacement component must be the same, meaning that :ref:`passive
components <components/passive-components:Passive components>` can only be swapped for other passive
components, and :ref:`op-amps <components/op-amps:Op-amps>` can only be swapped for other op-amps.
