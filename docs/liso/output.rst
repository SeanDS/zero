.. include:: /defs.txt

LISO output file parsing
========================

Known incompatibilities
-----------------------

Duplicate component and node names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In LISO, nodes and components may share the same name, and the user is warned
that this may lead to confusion. In |Zero|, nodes and components cannot share
the same name.

Output coordinates
~~~~~~~~~~~~~~~~~~

The parser assumes all outputs are in `db` and `degrees` (noise columns are
handled appropriately, however). This leads to incorrect results. This is
easily fixed but not yet implemented.

Differences in behaviour
------------------------

Input noise sinks
~~~~~~~~~~~~~~~~~

In LISO, input noise is always specified at the input `node`, and not the input `component`, even if
the circuit input is a current (i.e. ``iinput``). This makes no difference to the computed spectra,
but it does influence the labels used to plot the data. In |Zero| simulations, and in parsed LISO
output files, the input noise sink is always set to whatever the circuit input actually is - either
the input node in the case of ``uinput`` or the input component in the case of ``iinput``.
