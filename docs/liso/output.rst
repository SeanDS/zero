LISO output file parsing
========================

Known incompatibilities
-----------------------

Output coordinates
~~~~~~~~~~~~~~~~~~

The parser assumes all outputs are in `db` and `degrees` (noise columns are
handled appropriately, however). This leads to incorrect results. This is
easily fixed but not yet implemented.
