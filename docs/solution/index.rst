Solutions
=========

.. code-block:: python

   >>> from zero.solution import Solution

The :class:`.Solution` class provides a mechanism for storing, displaying and saving
the output of an :ref:`analysis <analyses/index:Analyses>`.

Combining solutions
-------------------

Solutions from different analyses can be combined and plotted together. The method :meth:`.Solution.combine`
takes as an argument another solution, and returns a new solution containing functions from both.


.. warning::

    In order to be combined, the solutions must have identical frequency vectors, but *no* identical
    functions.

Here is an example using a :ref:`LISO model <liso/index:LISO compatibility>` of an RF summing box
with two inputs and one output:

.. code-block:: python

    from zero.liso import LisoInputParser

    # create parser
    parser = LisoInputParser()

    base_circuit = """
    l l2 420n nlf nout
    c c4 47p nlf nout
    c c1 1n nrf gnd
    r r1 1k nrf gnd
    l l1 600n nrf n_l1_c2
    c c2 330p n_l1_c2 n_c2_c3
    c c3 33p n_c2_c3 nout
    c load 20p nout gnd

    freq log 100k 100M 1000
    uoutput nout
    """

    # parse base circuit
    parser.parse(base_circuit)
    # set input to low frequency port
    parser.parse("uinput nlf 50")
    # ground unused input
    parser.parse("r nrfsrc 5 nrf gnd")
    # calculate solution
    solutionlf = parser.solution()
    solutionlf.name = "LF Circuit"

    # reset parser state
    parser.reset()

    # parse base circuit
    parser.parse(base_circuit)
    # set input to radio frequency port
    parser.parse("uinput nrf 50")
    # ground unused input
    parser.parse("r nlfsrc 5 nlf gnd")
    # calculate solution
    solutionrf = parser.solution()
    solutionrf.name = "RF Circuit"

    # combine solutions
    solution = solutionlf.combine(solutionrf)

    # plot
    solution.plot()
    solution.show()

.. image:: /_static/solution-combination.svg

.. hint::

    Where solutions containing incompatible results are combined, such as with :ref:`signal <analyses/ac/signal:Small AC signal analysis>`
    and :ref:`noise <analyses/ac/noise:Small AC noise analysis>` analyses, the functions are combined
    but plotted separately.
