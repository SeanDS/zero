.. include:: /defs.txt

####################
Op-amp library tools
####################

|Zero|'s command line interface can be used to search the :class:`op-amp <.OpAmp>`
library bundled with the project.

Search queries
--------------

Search queries are specified as a set of declarative filters after the ``zero opamp``
command. |Zero| implements an expression parser which allows queries to be
arbitrarily long and complex, e.g.:

.. code-block:: text

    $ zero opamp "model != OP27 & ((vnoise <= 2n & vcorner < 10) | (vnoise <= 25n & inoise < 100f & icorner < 100))" --vnoise --vcorner --inoise --icorner

    ╒═════════╤════════════════════╤════════════╤════════════════════╤════════════╕
    │ Model   │ vnoise             │ vcorner    │ inoise             │ icorner    │
    ╞═════════╪════════════════════╪════════════╪════════════════════╪════════════╡
    │ OPA671  │ 10.000 nV/sqrt(Hz) │ 1.0000 kHz │ 2.0000 fA/sqrt(Hz) │ 2.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ OPA657  │ 4.8000 nV/sqrt(Hz) │ 2.0000 kHz │ 1.3000 fA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ OP00    │ 0.0000 V/sqrt(Hz)  │ 1.0000 Hz  │ 0.0000 A/sqrt(Hz)  │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ PZTFET1 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ OPA2604 │ 10.000 nV/sqrt(Hz) │ 200.00 Hz  │ 6.0000 fA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ PZTFET2 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ LT1028  │ 850.00 pV/sqrt(Hz) │ 3.5000 Hz  │ 1.0000 pA/sqrt(Hz) │ 250.00 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ OPA604  │ 10.000 nV/sqrt(Hz) │ 200.00 Hz  │ 6.0000 fA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ PZTFET3 │ 1.0000 nV/sqrt(Hz) │ 1.0000 Hz  │ 1.0000 pA/sqrt(Hz) │ 1.0000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ AD706   │ 17.000 nV/sqrt(Hz) │ 3.0000 Hz  │ 50.000 fA/sqrt(Hz) │ 10.000 Hz  │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ AD8628  │ 22.000 nV/sqrt(Hz) │ 1.0000 µHz │ 5.0000 fA/sqrt(Hz) │ 1.0000 µHz │
    ├─────────┼────────────────────┼────────────┼────────────────────┼────────────┤
    │ OPA655  │ 6.0000 nV/sqrt(Hz) │ 5.0000 kHz │ 1.3000 fA/sqrt(Hz) │ 1.0000 Hz  │
    ╘═════════╧════════════════════╧════════════╧════════════════════╧════════════╛

The expression must be defined on one line. Whitespace is ignored. Where values are specified,
such as "1n", these are parsed by :class:`.Quantity`
(see :ref:`Formatting and parsing quantities <format/index:Formatting and parsing quantities>`).

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

``=``
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

.. click:: zero.__main__:opamp
   :prog: zero opamp
   :show-nested:
