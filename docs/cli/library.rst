.. include:: /defs.txt

#######################
Component library tools
#######################

|Zero|'s command line interface can be used to search the :class:`op-amp <.OpAmp>`
library bundled with the project.

Listing the user library file path
----------------------------------

The built-in op-amp definitions can be supplemented or overridden by a user-defined
op-amp library. This library is stored within the user's home directory in a location
that depends on the operating system.

The path to this file can be listed with the command ``zero library path``.

Creating a user library
-----------------------

An empty user library can be created with ``zero library create``.

Opening the user library for editing
------------------------------------

The user library can be opened with the command ``zero library edit``.

Removing the user library
-------------------------

The user library can be removed with ``zero library remove``.

Showing the library
-------------------

The combined contents of the built-in library and any user-defined additions or overrides can be
printed to the screen with ``zero library show``. For large libraries, it is often useful to
specify the ``--paged`` flag to allow the contents to be navigated.

Search queries
--------------

Search queries are specified as a set of declarative filters after the ``zero library search``
command. |Zero| implements an expression parser which allows queries to be
arbitrarily long and complex, e.g.:

.. command-output:: zero library search "model != OP* & ((vnoise <= 2n & vcorner < 10) | (vnoise <= 25n & inoise < 100f & icorner < 100))"

The expression must be defined on one line. Whitespace is ignored. Where values are specified,
such as "1n", these are parsed by :class:`.Quantity`
(see :ref:`Parsing and displaying quantities <format/index:Parsing and displaying quantities>`).

Where a string comparison is made, e.g. with ``model``, wildcards are supported:

``*``
  Match any number of characters (including zero), e.g. ``OP*`` would match ``OP27``, ``OP37``,
  ``OP227``, etc.
``?``
  Match a single character, e.g. ``LT1?28`` would match ``LT1028`` and ``LT1128`` but not
  ``LT10028``.

Available parameters
~~~~~~~~~~~~~~~~~~~~

The following op-amp library parameters can be searched:

``model``
  Model name, e.g. `OP27`.
``a0``
  Open loop gain.
``gbw``
  Gain-bandwidth product.
``delay``
  Delay.
``vnoise``
  Flat voltage noise.
``vcorner``
  Voltage noise corner frequency.
``inoise``
  Flat current noise.
``icorner``
  Current noise corner frequency.
``vmax``
  Maximum output voltage.
``imax``
  Maximum output current.
``sr``
  Slew rate.

Operators
~~~~~~~~~

Expressions can use the following operators:

``==``
  Equal.
``!=``
  Not equal.
``<``
  Less than.
``<=``
  Less than or equal.
``>``
  Greater than.
``>=``
  Greater than or equal.
``&``
  Logical AND.
``|``
  Logical OR.

Groups
~~~~~~

Parentheses may be used to delimit groups:

.. code-block:: text

    (vnoise < 10n & inoise < 10p) | (vnoise < 100n & inoise < 1p)

Display
~~~~~~~

The results are by default displayed in a table. The rows are sorted based on the order in which the
parameters are defined in the search query, from left to right, with the leftmost parameter being
sorted last. The default sort direction is defined based on the parameter. The sort direction can be
specified explicitly as ``ASC`` (ascending) or ``DESC`` (descending) with the corresponding
``--sort`` parameter:

==================  ===========  =================
Flag                Parameter    Default direction
==================  ===========  =================
``--sort-a0``       ``a0``       descending
``--sort-gbw``      ``gbw``      descending
``--sort-delay``    ``delay``    ascending
``--sort-vnoise``   ``vnoise``   ascending
``--sort-vcorner``  ``vcorner``  ascending
``--sort-inoise``   ``inoise``   ascending
``--sort-icorner``  ``icorner``  ascending
``--sort-vmax``     ``vmax``     descending
``--sort-imax``     ``imax``     descending
``--sort-sr``       ``sr``       ascending
==================  ===========  =================

Parameters that are not explicitly searched are not ordered.

The display of the results table can be disabled using the ``--no-show-table`` flag. The results
can also be saved into a text file by specifying it with ``--save-data``. The specified file
extension will be used to guess the format to use, e.g. `csv` for comma-separated values or `txt`
for tab-separated values.

Results can also be plotted. The flags ``--plot-voltage-noise``, ``--plot-current-noise`` and
``--plot-gain`` can be used to plot the voltage and current noise or open loop gain of the op-amp,
respectively. Generated plots can also be saved by specifying a filename (or multiple filenames,
if you like) with the ``--save-voltage-noise-figure``, ``--save-current-noise-figure`` and
``--save-gain-figure`` options, respectively. Figures can be saved without being displayed with
``--no-plot-voltage-noise``, ``--no-plot-current-noise`` and ``--no-plot-gain``, respectively.

The following command will produce the plot below.

.. code-block:: bash

  $ zero library search "gbw > 800M & ((vnoise < 10n & inoise < 10p) | (vnoise < 100n & inoise < 1p)) & model != OP00" --plot-gain --fstop 1M

.. image:: /_static/cli-opamp-gain.svg

Command reference
-----------------

.. click:: zero.__main__:library
   :prog: zero library
   :show-nested:
