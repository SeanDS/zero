.. include:: /defs.txt

#################################
Parsing and displaying quantities
#################################

.. code-block:: python

   >>> from zero.format import Quantity

|Zero| contains a powerful quantity parser and formatter, allowing you to specify and display
component values, frequencies and other values in a natural way without resorting to mathematical
notation or special formatting.

Parsing quantities
------------------

The quantity parser accepts SI units and prefixes:

.. code-block:: python

   >>> Quantity("1.23 kHz")
   1230.0
   >>> Quantity("1e-3 mF")
   1e-06

Mathematical notation and SI prefixes can be combined arbitrarily, and spaces are ignored.

Displaying quantities
---------------------

A quantity's unit travels with its value. You can print a quantity with its value by calling for its
string representation:

.. code-block:: python

   >>> str(Quantity("45.678MHz"))
   '45.678 MHz'

An appropriate prefix is chosen even if the quantity is instantiated with another prefix:

.. code-block:: python

   >>> str(Quantity("45678 kHz"))
   '45.678 MHz'

More control can be gained over the display of the quantity by calling the :meth:`~.format.Quantity.format`
method. This allows units to be hidden, scale factors to be removed, and for a custom precision to
be specified.

Mathematical operations with quantities
---------------------------------------

For all intents and purposes, quantities behave like :class:`floats <float>`. That means you can
multiply, divide, add and subtract them from other quantities or scalars, and use them with standard
Python code.

Units are not carried over to the results of mathematical operations; the results of operations
involving quantities are always :class:`floats <float>`. Unit propagation is beyond the scope of
|Zero|.
