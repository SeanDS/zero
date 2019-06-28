Solutions
=========

.. code-block:: python

   >>> from zero.solution import Solution

The :class:`.Solution` class provides a mechanism for storing, displaying and saving
the output of an :ref:`analysis <analyses/index:Analyses>`; these are usually
:ref:`responses <data/index:Responses>` and :ref:`noise spectral densities <data/index:Noise spectral densities>`.

Retrieving functions
--------------------

Using :meth:`.filter_responses` and :meth:`.filter_noise`, functions can be retrieved by matching
against sources, sinks, groups and, in the case of noise, types. These methods return a
:class:`dict` containing the matched functions in lists keyed by their group names. To retrieve an
individual function, two convenience methods are provided: :meth:`.get_response` and
:meth:`~.Solution.get_noise`. These take as arguments the source and sink of the :class:`~.data.Response`
or :class:`~.data.NoiseDensity` to retrieve. The source and sink in :meth:`.get_response` and the
sink in :meth:`~.Solution.get_noise` can be :class:`components <.Component>`
or :class:`nodes <.Node>` or names, while the source in :meth:`~.Solution.get_noise` can be a
:class:`~.components.Noise` or noise specifier such as ``V(op1)``
(:ref:`see below <solution/index:Specifying noise sources and sinks>`).

Specifying response sources and sinks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~.data.Response` sources and sinks (and :class:`~.components.Noise` sinks) specified in
:meth:`~.Solution.get_noise` are always components or nodes. You can specify these using either the
corresponding :class:`.Component` or :class:`.Node` objects or by specifying their name as a
string.

Assuming that a circuit is built in the following way...

.. code-block:: python

   >>> import numpy as np
   >>> from zero import Circuit
   >>> from zero.analysis import AcSignalAnalysis
   >>> circuit = Circuit()
   >>> circuit.add_opamp(name="op1", model="OP27", node1="gnd", node2="nin", node3="nout")
   >>> circuit.add_resistor(name="r1", value="1k", node1="nin", node2="nout")
   >>> signal_analysis = AcSignalAnalysis(circuit)
   >>> solution = signal_analysis.calculate(frequencies=np.logspace(0, 4, 1001), input_type="voltage", node="nin")

...responses between the input node and various nodes and components can be retrieved in the
following ways:

.. code-block:: python

   >>> nin = circuit["nin"] # get the input node object
   >>> nout = circuit["nout"] # get the output node object
   >>> print(solution.get_response(nin, nout)) # response between input and output nodes
   nin to nout (V/V)
   >>> print(solution.get_response("nin", "nout")) # alternative string specifier
   nin to nout (V/V)
   >>> print(solution.get_response("nin", "r1")) # response between input node and resistor current (note the units)
   n1 to r1 (A/V)

Specifying noise sources and sinks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to retrieve a noise function from a solution, you must specify the noise source in
:meth:`~.Solution.get_noise`. Noise sources can either be specified using their
:class:`noise object <.components.Noise>` or by building a noise specifier string. Noise sinks are
specified in the same way as response sinks (:ref:`see above <solution/index:Specifying response sources and sinks>`).

Specifying the noise source by its object involves first retrieving the component that produces the
noise. Each component holds its noise sources in its :ref:`properties <Components/index:Component noise sources>`.
For example, op-amps have voltage noise at their output and current noise at their inverting and
non-inverting inputs. Assuming the op-amp is referenced by ``op1``, these can be retrieved using
``op1.voltage_noise``, ``op1.inv_current_noise`` and ``op1.non_inv_current_noise``, respectively.

An alternative approach is to use a noise specifier string. These are strings constructed in the
form ``prefix(component-name[, node-name])``, with the prefix representing the type of noise as
shown in this table:

============================  ======  ==============
Noise type                    Prefix  Example
============================  ======  ==============
Resistor (Johnson)            ``R``   ``R(r1)``
Op-amp voltage                ``V``   ``V(op1)``
Op-amp non-inverting current  ``I``   ``I(op1, np)``
Op-amp inverting current      ``I``   ``I(op1, nm)``
============================  ======  ==============

Assuming that a circuit is built in the following way...

.. code-block:: python

   >>> import numpy as np
   >>> from zero import Circuit
   >>> from zero.analysis import AcNoiseAnalysis
   >>> circuit = Circuit()
   >>> circuit.add_opamp(name="op1", model="OP27", node1="gnd", node2="nin", node3="nout")
   >>> circuit.add_resistor(name="r1", value="1k", node1="nin", node2="nout")
   >>> noise_analysis = AcNoiseAnalysis(circuit)
   >>> solution = noise_analysis.calculate(frequencies=np.logspace(0, 4, 1001), input_type="voltage", node="nin", sink="nout")

...noise functions can be retrieved with e.g.:

.. code-block:: python

   >>> op1 = circuit["op1"] # get the op1 object
   >>> print(solution.get_noise(op1.voltage_noise, "nout")) # voltage noise at op1
   V(op1) to nout
   >>> print(solution.get_noise("V(op1)", "nout")) # alternative string specifier
   V(op1) to nout
   >>> print(solution.get_noise(op1.non_inv_current_noise, "nout")) # current noise at op1's non-inverting input
   I(op1, nin) to nout
   >>> print(solution.get_noise("I(op1, nin", "nout")) # alternative string specifier

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
