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

.. code-block:: text

    $ zero library search "model != OP* & ((vnoise <= 2n & vcorner < 10) | (vnoise <= 25n & inoise < 100f & icorner < 100))" --vnoise --vcorner --inoise --icorner

    ╒═════════╤════════════════════╤════════════╤════════════════════╤════════════╕
    │ Model   │ vnoise             │ vcorner    │ inoise             │ icorner    │
    ╞═════════╪════════════════════╪════════════╪════════════════════╪════════════╡
    │ PZTFET1 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ PZTFET2 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ LT1028  │ 850.00 pV/sqrt(Hz) │ 3.5000 Hz  │ 1.0000 pA/sqrt(Hz) │ 250.00 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ PZTFET3 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ AD706   │ 17.000 nV/sqrt(Hz) │ 3.0000 Hz  │ 50.000 fA/sqrt(Hz) │ 10.000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ AD8628  │ 22.000 nV/sqrt(Hz) │ 1.0000 µHz │ 5.0000 fA/sqrt(Hz) │ 1.0000 µHz │
    ╘═════════╧════════════════════╧════════════╧════════════════════╧════════════╛

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

The results are displayed in a table. By default, only the op-amp model names
matching a given query are displayed in the table. To add extra columns,
specify the corresponding flag as part of the call:

``--a0``
  Show open loop gain.
``--gbw``
  Show gain-bandwidth product.
``--delay``
  Show delay.
``--vnoise``
  Show flat voltage noise.
``--vcorner``
  Show voltage noise corner frequency.
``--inoise``
  Show flat current noise.
``--icorner``
  Show current noise corner frequency.
``--vmax``
  Show maximum output voltage.
``--imax``
  Show maximum output current.
``--sr``
  Show slew rate.

Command reference
-----------------

.. click:: zero.__main__:library
   :prog: zero library
   :show-nested:
