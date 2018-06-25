LISO input file parsing
=======================

Known incompatibilities
-----------------------

LISO Perl commands
~~~~~~~~~~~~~~~~~~

Commands used for running LISO in a loop with `pfil` are not supported. Instead you
can use `circuit` as part of a Python script to run either LISO or native `circuit`
simulations in a loop.

Known differences
-----------------

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
