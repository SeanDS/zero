Circuit documentation
=====================

.. note:: `Circuit` is still under construction, with the program structure and behaviour
        **not at all** stable. The program, as with this documentation, may be altered in
        ways that break existing scripts at any time without notice.

`Circuit` is a linear circuit simulation library and command line tool. It is able to
perform small signal AC analysis on collections of components such as resistors, capacitors,
inductors and op-amps to predict transfer functions and noise.

==============
Why `circuit`?
==============

Given that tools such as `LTspice <http://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`_
and `Qucs <http://qucs.sourceforge.net/>`_ exist, why use this tool?

The answer is: it depends. For simple circuits where precision is not critical, or where you must
model non-linear or time-variant effects, then the above tools are potentially useful; however,
whilst manufacturers often provide SPICE models to represent their parts, these often do not
correctly model noise, open loop gain and output impedance. :ref:`LISO <index:LISO>`, upon which
`circuit` is based, was motivated in part by this reason, and instead provided *measured* op-amp
data as part of its library, which became incredibly useful to its users.

===================
What `circuit` does
===================

`circuit` can perform small signal analyses on circuits containing linear components. It is
inherently ac, and as such can compute frequency responses between nodes or components and
noise spectral densities at nodes. Inputs and outputs can be specified in terms of voltage or
current (except noise, which, for the time being can only be computed as a voltage).

==========================
What `circuit` does not do
==========================

`circuit` linearises the circuit before performing an analysis, and as such distorsion, saturation
or intermodulation that would occur in the real circuit is not considered. It also assumes that the
circuit's components' operating points are around zero, and so non-linear components which can be
operated in a linear regime such as transistors cannot be simulated, nor the effect of dc offsets
on op-amps. One exception to this is the :class:`op-amp <.OpAmp>`, which is available in `circuit`
but only as a linearised component; as such, limits to the op-amp's input or output swing are
not able to be simulated in the standard analysis.

It is in principle possible to linearise non-linear components as a first step before performing an
ac analysis (i.e. to compute a non-zero operating point); this is not yet possible with `circuit`
but may be available in the future. For those wishing to simulate circuits containing non-linear
components, try a variety of SPICE (e.g. `LTspice <http://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`_)
or `Qucs <http://qucs.sourceforge.net/>`_.

====
LISO
====

`Circuit` is based on `LISO <https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso>`_
by Gerhard Heinzel. It (mostly) understands LISO circuit mode input files, meaning that it
can be used in place of LISO to simulate circuit signals. It also understands LISO output
files, allowing results previously computed with LISO to be plotted, and for LISO results
to be directly compared to those of this program.

Contents
========

.. toctree::
   :maxdepth: 2

   introduction/index
   circuit/index
   components/index
   analyses/index
   solution/index
   data/index
   format/index
   examples/index
   liso/index
   API reference <api/modules>
   contributing/index
