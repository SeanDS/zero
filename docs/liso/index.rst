.. include:: /defs.txt

LISO compatibility
==================

|Zero| somewhat understands `LISO <https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso>`_
input and output files. It is also capable of running a locally available LISO binary
and then plotting its results.

.. note::

    In order to solve a circuit, |Zero| implicitly calculates responses to all sinks or noise from
    all sources, depending on the type of analysis. LISO, however, only outputs the functions
    specified as outputs or noise sources in the script. Instead of throwing away this extra data,
    |Zero| stores all calculated functions in its :ref:`solution <solution/index:Solutions>`.
    In order for the produced plots to be identical to those of LISO, the functions requested in
    LISO are set as `default` in the solution such that they are plotted by :meth:`.Solution.plot`.
    The other functions, however, are still available to be plotted by calling
    :meth:`.Solution.plot_responses` or :meth:`.Solution.plot_noise` with appropriate arguments.

Parsing LISO files
------------------

.. toctree::
    :maxdepth: 2

    input
    output
