.. include:: /defs.txt

######################
Command line interface
######################

|Zero| provides a command line interface to perform some common tasks:

- :ref:`Run LISO scripts <cli/liso:LISO tools>`
- :ref:`Parametrically search the op-amp library <cli/opamp-library:Op-amp library tools>`
- :ref:`Download and display datasheets <cli/datasheets:Datasheets>`

===========
Subcommands
===========

.. toctree::
    :maxdepth: 2

    liso
    opamp-library
    datasheets

====================
Command line options
====================

.. click:: zero.__main__:cli
   :prog: zero
   :show-nested:
