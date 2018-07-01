AC analyses
===========

==============
Implementation
==============

-----------------
Circuit equations
-----------------

Components and nodes describe themselves using `Kirchoff's voltage and current
laws <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_. When
calculating signals or noise, the solver will ask each
:class:`component <.Component>` or :class:`node <.Node>`
for its :class:`equation <.BaseEquation>`, which describes the
potential differences the component creates between nodes and the corresponding
current `admittance <https://en.wikipedia.org/wiki/Admittance>`_ that is created.

.. toctree::
    :maxdepth: 2

    signal
    noise
