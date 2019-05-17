.. include:: /defs.txt

|Zero| documentation
====================

.. note::

   |Zero| is still under construction, with the program structure and behaviour **not** stable.
   The program, and this documentation, may be altered in ways that break existing scripts at any
   time without notice.

|Zero| is a linear circuit simulation library and command line tool. It is able to
perform small signal AC analysis on collections of components such as resistors, capacitors,
inductors and op-amps to predict responses and noise.

===========
Why |Zero|?
===========

Given that tools such as `LTspice <http://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`_
and `Qucs <http://qucs.sourceforge.net/>`_ exist, why use this tool?

The answer is: it depends. For simple circuits where precision is not critical, or where you must
model non-linear or time-variant effects, then the above tools are potentially useful; however,
whilst manufacturers often provide SPICE models to represent their parts, these often do not
correctly model noise, open loop gain and output impedance. :ref:`LISO <index:LISO>`, upon which
|Zero| is based, was motivated in part by this reason, and instead provided *measured* op-amp
data as part of its library, which became incredibly useful to its users.

================
What |Zero| does
================

|Zero| can perform small signal analyses on circuits containing linear components. It is
inherently AC, and as such can compute :ref:`frequency responses between nodes or components <analyses/ac/signal:Small AC signal analysis>`
and :ref:`noise spectral densities at nodes <analyses/ac/noise:Small AC noise analysis>`. Inputs and
outputs can be specified in terms of voltage or current (except noise, which, for the time being can
only be computed as a voltage).

For more information, see :ref:`the available AC analyses <analyses/ac/index:Available analyses>`.

==========================
What |Zero| does not do
==========================

|Zero| can only analyse linear time invariant (LTI) circuits. This means that the parameters of the
components within the circuit cannot change over time, so for example the charging of a capacitor or
the switching of a MOSFET cannot be simulated. This rules out certain simulations, such as
those of switch-mode power supply circuits and power-on characteristics, and also effects such as
distorsion, saturation and intermodulation that would appear in real circuits. Instead, the circuit
is assumed to be at its operating point, and the circuit is linearised around zero, such that if the
current through a component is reversed, the voltage drop across that component is also reversed.

It is in principle possible to linearise non-linear components as a first step before performing an
ac analysis (i.e. to compute a non-zero operating point); this is not yet possible with |Zero|
but may be available in the future. For those wishing to simulate circuits containing non-linear
components, try a variety of SPICE (e.g. `LTspice <http://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`_)
or `Qucs <http://qucs.sourceforge.net/>`_.

For more information, see :ref:`analyses/ac/index:AC analyses`.

====
LISO
====

|Zero| is based on `LISO <https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso>`_
by Gerhard Heinzel. It (mostly) understands LISO circuit mode input files, meaning that it
can be used in place of LISO to simulate circuit signals. It also understands LISO output
files, allowing results previously computed with LISO to be plotted, and for LISO results
to be directly compared to those of this program.

Contents
========

.. toctree::
   :maxdepth: 3

   introduction/index
   circuit/index
   components/index
   analyses/index
   solution/index
   data/index
   format/index
   examples/index
   liso/index
   cli/index
   configuration/index
   developers/index
   contributing/index
