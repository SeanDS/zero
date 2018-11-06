Analyses
========

.. code-block:: python

   >>> from zero.analysis import AcSignalAnalysis, AcNoiseAnalysis

The circuit can be solved in order to compute transfer functions or noise in a
single run. If both transfer functions *and* noise are required, then these
must be obtained in separate calls.

Transfer functions between the circuit input and an output (or outputs) can be
computed with :class:`.AcSignalAnalysis`. Noise from components and nodes in the
circuit at a particular node can be calculated with :class:`.AcNoiseAnalysis`.

.. toctree::
    :maxdepth: 2

    ac/index
