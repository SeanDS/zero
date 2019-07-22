.. include:: /defs.txt

|Zero| documentation
====================

.. note::

   |Zero| is still under construction, with the program structure and behaviour **not** stable.
   The program, and this documentation, may be altered in ways that break existing scripts at any
   time without notice.

|Zero| is a linear circuit simulation library and command line tool. It is able to
perform small signal ac analysis on collections of components such as resistors, capacitors,
inductors and op-amps to predict responses and noise.

===========
Why |Zero|?
===========

Given that tools such as `LTspice
<http://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html>`_ and
`Qucs <http://qucs.sourceforge.net/>`_ exist, why use this tool?

The answer is: `it depends`. For circuits where dc analysis is needed, or where you must
model non-linear or time-variant effects, then the above tools are very useful; however,
whilst component manufacturers often provide SPICE models to represent their parts, these often do
not correctly model noise and output impedance. This is especially true for op-amps, at
least historically. One of the key advantages of :ref:`LISO <index:LISO>`, upon which |Zero| is
loosely based, was that *measured* op-amp parameters were provided as standard in its library,
available to be used in simulations to provide accurate gain and noise results. This feature has
become incredibly useful to LISO's users. LISO furthermore provided an optimiser to be able to use
to tune circuit component values, something which is much trickier to do with SPICE.

|Zero| implements around half of what LISO is capable of doing, but extends it in a few ways to
provide greater customisability and ease of post-processing. You can for example easily add new
noise sources to components and simulate how they propagate through circuits without having to edit
any core code, or implement your own types of analysis using the base |Zero| circuit objects. The
results are also provided as a so-called `solution` which contains the simulation data as well as
means to plot, export and further process it.

The other half of LISO is the so-called `root` mode, which includes a powerful optimiser. This is
not part of |Zero|, but given that |Zero| exists within the Python ecosystem is is possible to use
it some other optimisation packages such as `scipy.optimize`. LISO's optimiser may one day be
implemented in |Zero|.

================
What |Zero| does
================

|Zero| can perform small signal analyses on circuits containing linear components. It is inherently
AC, and as such can compute :ref:`frequency responses between nodes or components
<analyses/ac/signal:Small AC signal analysis>` and :ref:`noise spectral densities at nodes
<analyses/ac/noise:Small AC noise analysis>`. Inputs and outputs can be specified in terms of
voltage or current.

For more information, see :ref:`the available AC analyses <analyses/ac/index:Available analyses>`.

==========================
What |Zero| does not do
==========================

|Zero|'s scope is fairly limited to the problem of simple op-amp circuits in the frequency domain
using linear time invariant (LTI) analysis. This means that the parameters of the components within
the circuit cannot change over time, so for example the charging of a capacitor or the switching of
a transistor cannot be simulated. This rules out certain simulations, such as those of switch-mode
power supply circuits and power-on characteristics, and also effects such as distorsion, saturation
and intermodulation that would appear in real circuits. Effects at dc such as op-amp input offset
voltages and currents are also not modelled. Instead, the circuit is assumed to be at its operating
point, and the circuit is linearised around zero, such that if the current through a component is
reversed, the voltage drop across that component is also reversed. A small signal analysis is then
performed to simulate the effect of the circuit's output given small variations in its input. This
is perfect for computing transfer functions and ac noise, but not for non-linear, time-varying
effects. You should bear these points in mind before choosing to use |Zero| for more complete
analyses.

For more information, see :ref:`analyses/ac/index:AC analyses`.

====
LISO
====

|Zero| is loosely based on `LISO
<https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso>`_ by Gerhard Heinzel. It
(mostly) understands LISO circuit mode input files, meaning that it can be used in place of LISO to
simulate circuit signals. It also understands LISO output files, allowing results previously
computed with LISO to be plotted, and for LISO results to be directly compared to those of this
program.

Contents
========

.. toctree::
   :maxdepth: 3

   introduction/index
   circuit/index
   components/index
   analyses/index
   solution/index
   plotting/index
   data/index
   format/index
   examples/index
   liso/index
   cli/index
   configuration/index
   developers/index
   contributing/index
