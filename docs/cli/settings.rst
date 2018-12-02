.. include:: /defs.txt

########
Settings
########

|Zero|'s command line interface can be used to create, edit, remove and list the settings file.

Listing the user configuration file path
----------------------------------------

The default settings can be supplemented or overridden by a user-defined configuration file. This
file is stored within the user's home directory in a location that depends on the operating system.

The path to this file can be listed with the command ``zero config path``.

Creating a user configuration
-----------------------------

An empty user configuration can be created with ``zero config create``.

Opening the user configuration for editing
------------------------------------------

The user configuration can be opened with the command ``zero config edit``.

Removing the user configuration
-------------------------------

The user library can be removed with ``zero library remove``.

Showing the configuration
-------------------------

The combined contents of the built-in configuration and any user-defined additions or overrides can
be printed to the screen with ``zero config show``. For large configurations, it is often useful to
specify the ``--paged`` flag to allow the contents to be navigated.

Styling plots
-------------

Plots generated with Matplotlib can have their style overridden by specifying commands in the
``plot.matplotlib`` section. For example, to specify the default line thickness, use the following
configuration::

    plot:
      matplotlib:
        lines.linewidth: 3

Refer to `this Matplotlib sample configuration file <https://matplotlib.org/users/customizing.html#a-sample-matplotlibrc-file>`_
for more configuration parameters.

Command reference
-----------------

.. click:: zero.__main__:config
   :prog: zero config
   :show-nested:
