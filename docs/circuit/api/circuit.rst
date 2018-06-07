.. currentmodule:: circuit.circuit

####################
The :class:`Circuit`
####################

.. code-block:: python

   >>> from circuit.circuit import Circuit

====================
What is a 'circuit'?
====================

A :class:`circuit <Circuit>` describes a collection of
:class:`components <.components.Component>` connected at
:class:`nodes <.components.Node>`. It may contain
:class:`resistors <.components.Resistor>`,
:class:`capacitors <.components.Capacitor>`,
:class:`inductors <.components.Inductor>` and
:class:`op-amps <.components.OpAmp>`, and the circuit can be supplied with an
:class:`input <.components.Input>` in order to produce a current through and
voltage across these components.

-------------------
Solving the circuit
-------------------

The circuit can be solved in order to compute transfer functions or noise in a
single run. If both transfer functions *and* noise are required, then these
must be obtained in separate runs.

Transfer functions between the circuit input and an output (or outputs) can be
computed with :meth:`.calculate_tfs`. Noise from components and nodes in the
circuit at a particular node can be calculated with :meth:`.calculate_noise`.

==============
Implementation
==============

-----------------
Circuit equations
-----------------

Components and nodes describe themselves using `Kirchoff's voltage and current
laws <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_. When
calculating signals or noise, the solver will ask each
:class:`component <.components.Component>` or :class:`node <.components.Node>`
for its :class:`equation <.components.BaseEquation>`, which describes the
potential differences the component creates between nodes and the corresponding
current `admittance <https://en.wikipedia.org/wiki/Admittance>`_ that is created.

===============
Class reference
===============

This reference contains the following `class` entries:

.. autosummary::
   :nosignatures:

   Circuit

-------
Details
-------

.. autoclass:: Circuit
    :members:
