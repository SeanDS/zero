.. include:: /defs.txt

AC analyses
===========

The available AC analyses are performed assuming the circuit to be linear time invariant (LTI),
meaning that the parameters of the components within the circuit, and the circuit itself, cannot
change over time. This is usually a reasonable assumption to make for circuits containing passive
components such as resistors, capacitors and inductors, and op-amps far from saturation, and where
only the frequency response or noise spectral density is required to be computed.

The linearity property implies the principle of superposition, such that if you double the circuit's
input voltage, the current through each component will also double. This restriction implies
that components with a non-linear relationship between current and voltage cannot be simulated.
Components such as diodes and transistors fall within this category, but it also means that certain
component properties such as the output swing limit of op-amps cannot be simulated. In certain
circumstances it is possible to approximate a linear relationship between a component's voltage and
current around the operating point, allowing the component to be simulated in the LTI regime, and
this property is exploited in the case of :class:`op-amps <.OpAmp>`.

The small-signal AC response of a circuit input to any of its components or nodes can be computed
with the :ref:`AC small signal analysis <analyses/ac/signal:Small AC signal analysis>`.
The noise spectral density at a node arising from components and nodes elsewhere in the circuit can
be computed using the :ref:`AC small signal noise analysis <analyses/ac/noise:Small AC noise analysis>`.

.. toctree::
    :maxdepth: 2

    signal
    noise

Implementation
##############

|Zero| uses a `modified nodal analysis <https://en.wikipedia.org/wiki/Modified_nodal_analysis>`_ to
compute circuit voltages and currents. Standard nodal analysis only allows voltages to be determined
across components by using `Kirchoff's current law <https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws#Kirchhoff's_current_law_(KCL)https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws>`_
(which states that the sum of all currents flowing into or out of each node equals zero). A series
of equations representing the inverse impedance of each component are built into a so-called
`admittance` matrix, which is then solved to find the voltage drop across each component. This works
well for circuits containing only passive components, but not those containing voltage sources.
Ideal voltage sources considered in |Zero| have by definition no internal resistance (in other
words, they can supply infinitely high current), and this makes the matrix representation of
such a circuit singular due to the row corresponding to the voltage source effectively
stating that ``0 = 1``! To handle voltage sources as well as other voltage-producing components like
op-amps, |Zero| extends the equations derived from Kirchoff's current law to include columns
representing voltage drops between nodes. The circuit can then be solved for a number of unknown
voltages and currents.

Consider the following voltage divider, defined in :ref:`LISO syntax <liso/input:LISO input file parsing>`:

.. code-block:: text

    r r1 1k n1 n2
    r r2 2k n2 gnd

    freq log 1 100 101

    uinput n1 0
    uoutput n2

The following circuit matrix is generated for this circuit (using
``zero liso /path/to/script.fil --print-matrix``):

.. code-block:: text

    ╒═══════╤═════════╤═════════╤════════════╤═════════╤═════════╤═══════╕
    │       │ i[r1]   │ i[r2]   │ i[input]   │ V[n2]   │ V[n1]   │ RHS   │
    ╞═══════╪═════════╪═════════╪════════════╪═════════╪═════════╪═══════╡
    │ r1    │ 1.00e3  │ ---     │ ---        │ 1       │ -1      │ ---   │
    ├───────┼─────────┼─────────┼────────────┼─────────┼─────────┼───────┤
    │ r2    │ ---     │ 2.00e3  │ ---        │ -1      │ ---     │ ---   │
    ├───────┼─────────┼─────────┼────────────┼─────────┼─────────┼───────┤
    │ input │ ---     │ ---     │ ---        │ ---     │ 1       │ 1     │
    ├───────┼─────────┼─────────┼────────────┼─────────┼─────────┼───────┤
    │ n2    │ 1       │ -1      │ ---        │ ---     │ ---     │ ---   │
    ├───────┼─────────┼─────────┼────────────┼─────────┼─────────┼───────┤
    │ n1    │ -1      │ ---     │ 1          │ ---     │ ---     │ ---   │
    ╘═══════╧═════════╧═════════╧════════════╧═════════╧═════════╧═══════╛

The entries containing ``---`` represent zero in sparse matrix form. The equations this matrix
represents look like this (using ``zero liso /path/to/script.fil --print-equations``):

.. code-block:: text

    1.00 × 10 ^ 3 × I[r1] + V[n2] - V[n1] = 0
            2.00 × 10 ^ 3 × I[r2] - V[n2] = 0
                                    V[n1] = 1
                            I[r1] - I[r2] = 0
                       - I[r1] + I[input] = 0

In the matrix, this equation is arranged to equal ``0``
on the right hand side; however, we can rearrange the equation for ``r1`` to read: "voltage drop
across ``r1`` equals the voltage difference between nodes ``n2`` and ``n1``". The equation for
``r2`` is similar, but since one of its nodes is attached to ground (which has ``0`` potential), the
voltage difference between its nodes is simply equal to the potential at ``n2``.

.. note::
    In |Zero|, the circuit is solved under the assumption that the circuit's ground is ``0``, which
    allows the use of standard linear analysis techniques. One side-effect of this approach is that
    circuits cannot contain floating loops, i.e. loops without a defined ground connection. This
    is usually not important, but has an effect on e.g. transformer circuits with intermediate
    or isolated loops. If reasonable for the application, consider supplying a weak connection to
    ground using e.g. a large valued resistor.

    This property does *not* affect the ability to include floating voltage sources in circuits,
    as long as these are contained in grounded loops.

In the row corresponding to the voltage input, there is no voltage drop across the source (as per
its definition as an ideal source), and its right hand side equals ``1``. This means that the
circuit is solved with the constraint that the voltage between ``n1`` and ground must always be
``1``. The solver can adjust all of the other non-zero matrix elements in the left hand side until
this condition is met within some level of tolerance.

Available analyses
~~~~~~~~~~~~~~~~~~

Signals
.......

The solution ``x`` to the matrix equation ``Ax = b``, where ``A`` is the circuit matrix above and
``b`` is the right hand side vector, gives the current through each component and voltage at each
node.

Noise
.....

Noise analysis requires an essentially identical approach to building the circuit matrix, except
that the matrix is transposed and the right hand side is given a ``1`` in the row corresponding to
the chosen noise output node instead of the input. This results in the solution ``x`` in the matrix
equation ``Ax = b`` instead providing what amounts to the reverse responses between the component
and nodes in the circuit and the chosen noise output node. These reverse responses are as a last
step multiplied by the noise at each component and node to infer the noise at the noise output node.
