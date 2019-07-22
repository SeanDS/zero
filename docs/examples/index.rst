.. include:: /defs.txt

Examples
========

Basic circuits
--------------

Some basic circuit example scripts are provided in the |Zero| source directory, or in the
`development repository`_. These scripts can be run directly as long as |Zero| is installed and
available.

LISO
----

.. toctree::
    :maxdepth: 2

    liso-input

.. _development repository: https://git.ligo.org/sean-leavey/zero/tree/master/examples/native

Generating a circuit graph
--------------------------

You can plot the circuit's node network using Graphviz, if installed:

.. code:: python

    >>> from zero.display import NodeGraph
    >>> NodeGraph(circuit)

.. image:: /_static/liso-input-node-graph.svg

Graphviz almost always produces a messy representation, but it can sometimes be useful to help
spot simple mistakes in circuit definitions.
