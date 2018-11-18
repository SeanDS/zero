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
