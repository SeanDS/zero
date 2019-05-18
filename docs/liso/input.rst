.. include:: /defs.txt

LISO input file parsing
=======================

Known incompatibilities
-----------------------

Duplicate component and node names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In LISO, nodes and components may share the same name, and the user is warned
that this may lead to confusion. In |Zero|, nodes and components cannot share
the same name.

Outputs
~~~~~~~

|Zero| does not support the ``deg+`` or ``deg-`` output coordinates. Please use ``deg`` instead.
It also throws an error when a LISO script's ``ioutput`` or ``uoutput`` commands contain only a
phase coordinate, e.g.:

.. code-block:: text

    uoutput nout:deg

Such outputs could in principle be handled by |Zero|, but it would add complexity to the
:class:`Solution` and :class:`Series` classes that is not worth the effort given how rare
this type of output is. In order to use such scripts with |Zero|, simply add a magnitude
unit, e.g.

.. code-block:: text

    uoutput nout:db:deg

Root mode
~~~~~~~~~

|Zero| does not support LISO's root mode, meaning that the fitting tools provided in LISO for
responses and noise spectra are not replicated. It is suggested to instead use |Zero| with a Python
optimisation library such as `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.
Note that it is very important for circuit responses and noise fitting to use a well-suited
optimiser, particularly one that can fit in log space. LISO's fitting library performs very well for
this purpose.

Commands
~~~~~~~~

The following commands are not yet supported:

- :code:`factor` (input multiplicative factor)
- :code:`inputnoise` (circuit noise referred to input node)
- :code:`zin` (input impedance)
- :code:`opdiff` (plot op-amp input differential voltage)
- :code:`margin` (compute op-amp phase margin; replaced :code:`opstab` in LISO v1.78)
- :code:`sens` (print table of component sensitivities)

Here are some commands which will probably not be supported:

- other `max` or `min` based commands, e.g. :code:`maxinput`
- :code:`eagle` (produce EAGLE file)
- :code:`gnuterm`
- component :code:`C0805` (0805 capacitor with parasitic properties; not implemented in
  favour of grouping components together with macros)

Op-amp library
~~~~~~~~~~~~~~

LISO's op-amp library format is not supported, but the full LISO library is bundled
in |Zero|'s native format.

The op-amp library is implemented in a different format to that of LISO,
primarily for logistical reasons: Python contains a convenient :class:`~configparser.ConfigParser`
library which can read and write config files similar to Windows :code:`INI` files,
but in a slightly different format to LISO's op-amp library format. The main
difference is that in :class:`~configparser.ConfigParser` files, repeated terms are not allowed in
the same entry, so LISO's use of multiple ``pole`` or ``zero`` entries under an
op-amp are not supported. Instead, the library represents poles and zeros as
single line expressions of comma separated values:

.. code-block:: text

    [op177]
    ...
    poles = 7.53M 1.78, 1.66M # fitted from measurement
    ...

Furthermore, the library improves on that of LISO's by allowing an ``alias`` setting where you can
specify other op-amps with the same properties:

.. code-block:: text

    [ad712]
    ...
    aliases = ad711, ad713
    ...

Finally, the parameters ``un``, ``uc`` ``in`` and ``ic`` have been renamed ``vnoise``, ``vcorner``,
``inoise`` and ``icorner``, respectively.

Submissions of op-amp parameters to |Zero|'s library are strongly encouraged
(see :ref:`contributing/index:Op-amp library additions`).

LISO Perl commands
~~~~~~~~~~~~~~~~~~

Commands used for running LISO in a loop with :code:`pfil` are not supported. Instead you
can use |Zero| as part of a Python script to run either LISO or native |Zero|
simulations in a loop.

Differences in behaviour
------------------------

Command order
~~~~~~~~~~~~~

In LISO, the output must be specified *after* the components. In |Zero|, order is
irrelevant.

`Noisy` command
~~~~~~~~~~~~~~~

.. code-block:: text

    noisy all|allr|allop|noise-source [all|allr|allop|noise-source] ...

The LISO manual states in section 7.3 regarding the noise sources used to calculate the
:code:`sum` output:

    Note also that all noise sources that are included in the `noise` instruction, i.e.
    those that are plotted individually, are automatically considered "noisy", i.e.
    they are always included in the sum.

In LISO, if the :code:`sum` output is present but there is no :code:`noisy` command, the following
error is displayed:

.. code-block:: text

    *** Error: No noisy components! (Try 'noisy all')

In |Zero|, the :code:`noisy` command does not need to be present as by default, even in LISO,
the noise sources that contribute to the :code:`sum` output always includes those specified in
the output itself. The :code:`noisy` command is available merely to add additional noise sources
to the :code:`sum` that are not explicitly plotted.

As the lack of presence of a :code:`noisy` command in this case does not yield *different*
results to LISO, only an error in one case and a reasonable output in the other, this
behaviour is not considered a bug.

String lengths
~~~~~~~~~~~~~~

LISO has a limit of 16 for most strings (component names, op-amp types, node names, etc.). In
|Zero| the limit is effectively arbitrary.

.. hint::
    In the case of *mutual inductance* commands, the name is entirely ignored. This is used in
    LISO only for fitting routines, which are not part of |Zero|.
