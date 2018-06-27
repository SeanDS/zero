Circuit documentation
=====================

.. note:: `Circuit` is still under construction, with the program structure and behaviour
        **not at all** stable. The program, as with this documentation, may be altered in
        ways that break existing scripts at any time without notice.

`Circuit` is a linear circuit simulation library and command line tool. It is able to
perform small signal AC analysis on collections of components such as resistors, capacitors,
inductors and op-amps to predict transfer functions and noise.

LISO
----

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
   components/index
   analyses/index
   api/index
   examples/index
   liso/index
   contributing/index
