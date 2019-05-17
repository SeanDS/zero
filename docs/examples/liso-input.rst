.. include:: /defs.txt

Input file parsing
==================

|Zero| is capable of parsing some LISO input files. First, import the LISO input parser:

.. code:: python

    from zero.liso import LisoInputParser

then create a parser object:

.. code:: python

    parser = LisoInputParser()

To parse a LISO circuit, either call the :meth:`~.LisoParser.parse` method with text:

.. code:: python

    parser.parse("""
    c c1 10u gnd n1
    r r1 430 n1 nm
    r r2 43k nm nout
    c c2 47p nm nout
    op o1 lt1124 nin nm nout

    freq log 1 100k 100

    uinput nin 0
    uoutput nout:db:deg
    """)

Or point it to a file using the :code:`path` parameter:

.. code:: python

    parser.parse(path="/path/to/liso/script.fil")

Get the solution with :meth:`~.LisoParser.solution` and plot and show it with
:meth:`.Solution.plot` and :meth:`.Solution.show`:

.. code:: python

    solution = parser.solution()
    solution.plot()
    solution.show()

.. image:: /_static/liso-input-response.svg

You can at any time list the circuit's constituent components:

.. code:: python

    parser.circuit

.. code-block:: text

    Circuit with 6 components and 5 nodes

    	1. c1 [in=gnd, out=n1, C=1e-05]
    	2. c2 [in=nm, out=nout, C=4.7e-11]
    	3. input [in=gnd, out=nin, Z=default]
    	4. o1 [in+=nin, in-=nm, out=nout, model=LT1124]
    	5. r1 [in=n1, out=nm, R=430.0]
    	6. r2 [in=nm, out=nout, R=43000.0]

You can also plot the circuit's node network using Graphviz, if installed:

.. code:: python

    from zero.display import NodeGraph
    NodeGraph(parser.circuit)

.. image:: /_static/liso-input-node-graph.svg

Graphviz almost always produces a messy representation, but it can sometimes be useful to help
spot simple mistakes in circuit definitions.
