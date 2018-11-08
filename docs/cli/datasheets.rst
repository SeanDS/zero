.. include:: /defs.txt

##########
Datasheets
##########

|Zero|'s command line interface can be used to download and display datasheets using
`Octopart <https://octopart.com/>`__'s API.

Searching for parts
-------------------

Specify a search term like this:

.. code-block:: bash

    $ zero datasheet "OP27"

Partial matches are made based on the search string by default. To force exact matches only,
specify ``--exact``.

If there are multiple parts found, a list is displayed and the user is prompted to enter a
number corresponding to the part to display. Once a part is selected, either its datasheet is
shown or, in the event that there are multiple datasheets available for the specified part, the
user is again prompted to choose a datasheet.

The selected datasheet is downloaded and displayed using the default viwer. To download the
datasheet without displaying it, use the ``--download-only`` flag and set the ``--path`` option to
the path to save the file. If no ``--path`` is specified, the datasheet is saved to a temporary
location and the location is printed to the screen.

To download and display the first part's latest datasheet, specify the ``--first`` flag, e.g.:

.. code-block:: bash

    $ zero datasheet "OP27" --first

This will immediately download and display the latest OP27 datasheet.

Updating the API endpoint and key
---------------------------------

|Zero| comes bundled with an API key which is open to use for any users. If for some reason this
API key is no longer available, a new key can be specified in the
:ref:`configuration file <configuration/index:Configuration>`.

Command reference
-----------------

.. click:: zero.__main__:datasheet
   :prog: zero datasheet
   :show-nested:
