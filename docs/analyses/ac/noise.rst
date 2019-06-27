.. currentmodule:: zero.analysis.ac.noise

Small AC noise analysis
=======================

Linear AC noise analysis.

Generating noise sums
---------------------

Incoherent noise sums can be created as part of the analysis and added to the :class:`.Solution`.
This is governed by the ``incoherent_sum`` parameter of :meth:`~.AcNoiseAnalysis.calculate`.

Setting ``incoherent_sum`` to ``True`` results in the incoherent sum of all noise in the circuit at
the specified noise sink being calculated and added as a single function to the solution.

Alternatively, ``incoherent_sum`` can be specified as a :class:`dict` containing legend labels as
keys and sequences of noise spectra as values. The noise spectra can either be
:class:`.NoiseDensity` objects or :ref:`noise specifier strings <solution/index:Specifying noise sources and sinks>`
as supported by :meth:`.Solution.get_noise`. The values may alternatively be the strings "all",
"allop" or "allr" to compute noise from all components, all op-amps and all resistors, respectively.

Sums are plotted in shades of grey determined by the plotting configuration's
``sum_greyscale_cycle_start``, ``sum_greyscale_cycle_stop`` and ``sum_greyscale_cycle_count``
values.

Examples
~~~~~~~~

Add a total incoherent sum to the solution:

.. code-block:: python

   solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1",
                                 sink="nout", incoherent_sum=True)

Add an incoherent sum of all resistor noise:

.. code-block:: python

   solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1",
                                 sink="nout", incoherent_sum={"resistors": "allr"})

Add incoherent sums of all resistor and op-amp noise:

.. code-block:: python

   # Shorthand syntax.
   solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1",
                                 sink="nout", incoherent_sum={"resistors": "allr",
                                                              "op-amps": "allop"})
   # Alternatively specify components directly using noise specifiers.
   solution = analysis.calculate(frequencies=frequencies, input_type="voltage", node="n1",
                                 sink="nout", incoherent_sum={"sum": ["R(r1)", "V(op1)"]})
