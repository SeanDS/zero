LISO input file parsing
=======================

Known incompatibilities
-----------------------

Outputs
~~~~~~~

`circuit` does not support the `deg+` or `deg-` output coordinates. Please use `deg` instead.
It also throws an error when a LISO script's `ioutput` or `uoutput` commands contain only a
phase coordinate, e.g.:

.. code-block:: text

    uoutput nout:deg

Such outputs could in principle be handled by `circuit`, but it would add complexity to the
:class:`Solution` and :class:`Series` classes that is not worth the effort given how rare
this type of output is. In order to use such scripts with `circuit`, simply add a magnitude
unit, e.g.

.. code-block:: text

    uoutput nout:db:deg

Root mode
~~~~~~~~~

`circuit` does not support LISO's root mode, meaning that fitting of transfer
functions is not available. It is suggested to instead use `circuit` with a Python
optimisation library such as `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

Commands
~~~~~~~~

The following commands are not yet supported:

- `factor` (input multiplicative factor)
- `m` (mutual inductance)
- `inputnoise` (circuit noise referred to input node)
- `zin` (input impedance)
- `opdiff` (plot op-amp input differential voltage)
- `margin` (compute op-amp phase margin; replaced `opstab` in LISO v1.78)
- `sens` (print table of component sensitivities)

Here are some commands which will probably not be supported:

- other `max` or `min` based commands, e.g. `maxinput`
- `eagle` (produce EAGLE file)    
- `gnuterm`
- component `C0805` (0805 capacitor with parasitic properties; not implemented in
  favour of grouping components together with macros)

Op-amp library
~~~~~~~~~~~~~~

LISO's op-amp library format is not supported, but the full LISO library is bundled
in `circuit`'s native format.

LISO Perl commands
~~~~~~~~~~~~~~~~~~

Commands used for running LISO in a loop with `pfil` are not supported. Instead you
can use `circuit` as part of a Python script to run either LISO or native `circuit`
simulations in a loop.

Differences in behaviour
------------------------

Command order
~~~~~~~~~~~~~

In LISO, the output must be specified *after* the components. In `circuit`, order is
irrelevant.

`Noisy` command
~~~~~~~~~~~~~~~

.. code-block:: text

    noisy all|allr|allop|noise-source [all|allr|allop|noise-source] ...

The LISO manual states in section 7.3 regarding the noise sources used to calculate the
`sum` output:

    Note also that all noise sources that are included in the `noise` instruction, i.e.
    those that are plotted individually, are automatically considered "noisy", i.e.
    they are always included in the sum.

In LISO, if the `sum` output is present but there is no `noisy` command, the following
error is displayed:

.. code-block:: text

    *** Error: No noisy components! (Try 'noisy all')

In `circuit`, the `noisy` command does not need to be present as by default, even in LISO,
the noise sources that contribute to the `sum` output always includes those specified in
the output itself. The `noisy` command is available merely to add additional noise sources
to the `sum` that are not explicitly plotted.

As the lack of presence of a `noisy` command in this case does not yield *different*
results to LISO, only an error in one case and a reasonable output in the other, this
behaviour is not considered a bug.

String lengths
~~~~~~~~~~~~~~

LISO has a limit of 16 for most strings (component names, op-amp types, node names, etc.). In `circuit` the
limit is effectively arbitrary.
