.. include:: /defs.txt
.. currentmodule:: zero.analysis.ac.noise

Small AC noise analysis
=======================

The small signal AC noise analysis calculates the :ref:`noise spectral densities <data/index:Noise
spectral densities>` at a particular :class:`.Node` or :class:`.Component` within a circuit due to
noise sources within the circuit, assuming that the noise is small enough not to influence the
operating point and gain of the circuit.

Generating noise sums
---------------------

Incoherent noise sums can be created as part of the analysis and added to the :class:`.Solution`.
This is governed by the ``incoherent_sum`` parameter of :meth:`~.AcNoiseAnalysis.calculate`.

Setting ``incoherent_sum`` to ``True`` results in the incoherent sum of all noise in the circuit at
the specified noise sink being calculated and added as a single function to the solution.

Alternatively, ``incoherent_sum`` can be specified as a :class:`dict` containing legend labels as
keys and sequences of noise spectra as values. The noise spectra can either be
:class:`.NoiseDensity` objects or :ref:`noise specifier strings <solution/index:Specifying noise
sources and sinks>` as supported by :meth:`.Solution.get_noise`. The values may alternatively be the
strings "all", "allop" or "allr" to compute noise from all components, all op-amps and all
resistors, respectively.

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

Referring noise to the input
----------------------------

It is often desirable to refer the noise at a node or component to the input. This is particularly
useful when modelling readout circuits (e.g. for photodetectors), where the input referred noise
shows the smallest equivalent signal spectral density that can be detected above the noise.

Noise analyses can refer noise at a node or component to the input by setting the ``input_refer``
flag to ``True`` in :meth:`~.AcNoiseAnalysis.calculate`, which makes |Zero| apply a response
function (from the noise sink to the input) to the noise computed at the noise sink. The resulting
noise has its ``sink`` property changed to the input. If ``input_type`` was set to ``voltage``, this
is the input node; whereas if ``input_type`` was set to ``current``, this is the input component.

.. note::

    The input referring response function is obtained by performing a separate :ref:`signal analysis
    <analyses/ac/signal:Small AC signal analysis>` with the same circuit as the noise analysis. The
    response from the input to the sink is then extracted and inverted to give the response from the
    sink to the input. The noise at the sink in the noise analysis is then multiplied by this input
    referring response function.
