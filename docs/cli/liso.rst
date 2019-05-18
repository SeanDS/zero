.. include:: /defs.txt

##########
LISO tools
##########

.. hint::
   Also see the documentation on :ref:`LISO compatibility <liso/index:LISO Compatibility>`.

|Zero| can parse LISO input and output files, run them natively or run them via a local
LISO binary and display the results. It can also compare its native results to that of
LISO by overlaying results in a plot or displaying a table of values.

Script path
-----------

For all calls to ``zero liso``, a script path (``FILE``) must be specified. This can either be a
LISO input or output file (commonly given ``.fil`` and ``.out`` extensions, respectively), and
|Zero| will choose an appropriate parser based on what it finds.

Verbose output
--------------

By default, the command line utility does not output any text except that which is requested.
Verbosity can be switched on with the ``-v`` flag. Specify ``-vv`` for greater verbosity.

.. note::

   The ``-v`` flag must be specified before the ``liso`` subcommand, i.e. ``zero -v liso [FILE]``.
   An error will occur if the flag is specified after a subcommand.

Simulating a LISO input script with |Zero|
------------------------------------------

LISO input scripts can be run natively with the ``zero liso`` command. The input file is first
parsed and then built into an :class:`analysis <.BaseAnalysis>` which is then solved.

.. code-block:: bash

    $ zero liso /path/to/liso/script.fil

The plotted functions specified in the LISO input file are reproduced in the default |Zero| plot,
including noise sums.

Re-simulating a LISO output file with |Zero|
--------------------------------------------

LISO result files contain a complete description of the simulated circuit, and as such can be
parsed by |Zero| and re-simulated natively.

.. code-block:: bash

    $ zero liso /path/to/liso/script.out

Simulating a LISO input script with an external LISO binary
-----------------------------------------------------------

|Zero| can simulate a LISO input script with a locally installed LISO binary using the ``--liso``
flag. |Zero| runs the script with LISO and then parses the output file so that you can take
advantage of its plotting capabilities.

The LISO binary path must be specified with the ``--liso-path`` option. This must point to the exact
binary file, not just its directory, but may be relative to the current directory.

.. code-block:: bash

    $ zero liso /path/to/liso/script.fil --liso --liso-path /path/to/liso/fil

An alternative is to set the ``LISO_PATH`` environment variable to point to the LISO binary. Since
LISO anyway requests that users set the ``LISO_DIR`` environment variable, on Unix systems this can
be used to set ``LISO_PATH`` either in the terminal profile (e.g. during the call with e.g.
``LISO_PATH=$LISO_DIR/fil_static zero liso ...``) or as part of the call:

.. code-block:: bash

    $ LISO_PATH=$LISO_DIR/fil_static zero liso /path/to/liso/script.fil --liso

.. warning::

   LISO uses a separate op-amp library to |Zero|, and these may differ if modifications have been
   made to one but not the other. Take care when comparing results between the two tools.

Comparing a native simulation to LISO
-------------------------------------

As |Zero| can simulate LISO input scripts both natively and using the LISO binary, it can also
overlay the results on one plot, or report the difference between the results textually.

To overlay the results in a plot, specify the ``--compare`` flag. |Zero| will then run the specified
input file itself and with LISO, then it will parse the LISO results and combine them with its own.
The resulting plot then contains each function, with the native results with solid lines and the
LISO results with dashed lines:

.. image:: /_static/liso-compare-response.svg

A textual representation of the differences can also be displayed by specifying ``--diff``. This
must be provided in addition to ``--compare``. When specified, this prints a table containing
the worst relative and absolute differences between the two solutions, and the frequencies at which
they occur:

.. code-block:: text

    ╒══════════════════╤═══════════════════════════════╤═══════════════════════════════╕
    │                  │ Worst difference (absolute)   │ Worst difference (relative)   │
    ╞══════════════════╪═══════════════════════════════╪═══════════════════════════════╡
    │ nin to op1 (A/V) │ 1.08e-11 (f = 316.23 kHz)     │ 9.78e-10 (f = 316.23 kHz)     │
    ├──────────────────┼───────────────────────────────┼───────────────────────────────┤
    │ nin to no (V/V)  │ 1.04e-08 (f = 79.433 kHz)     │ 9.54e-10 (f = 79.433 kHz)     │
    ╘══════════════════╧═══════════════════════════════╧═══════════════════════════════╛

Saving figures
--------------

Figures can be saved using the ``--save-figure`` option, which must be followed by a file path.
The format of the figure is controlled by the specified file extension. For example, save PNGs, PDFs
and SVGs with ``--save-figure response.png``, ``--save-figure response.pdf`` and ``--save-figure response.svg``,
respectively.

The ``--save-figure`` option can be specified multiple times to save multiple figures, e.g.:

.. code-block:: bash

    $ zero liso /path/to/liso/script.fil --save-figure response.png --save-figure response.pdf

Command reference
-----------------

.. click:: zero.__main__:liso
   :prog: zero liso
   :show-nested:
