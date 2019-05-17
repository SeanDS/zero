.. include:: /defs.txt

###############
Data containers
###############

.. code-block:: python

   >>> from zero.data import Response, NoiseDensity

|Zero| :ref:`analysis <analyses/index:Analyses>` results (responses and noise spectra) are
stored within `function` containers. These are relatively low level objects that hold each
function's data, its frequency axis, and any meta data produced by the analysis. These objects are
able to plot themselves when provided a figure to draw to. They also contain logic to compare
themselves to other functions, to check for equivalency.

In normal circumstances, you should not need to directly interact with these objects; rather, you
can plot and save their underlying data using a :ref:`Solution <solution/index:Solutions>`.

Responses
---------

:class:`Responses <.data.Response>` contain the response of a component or node to another component
or node. Each response contains references to the source and sink component or node, and its units.

Noise spectra
-------------

:class:`Noise spectra <.data.NoiseDensity>` contain the noise at a particular component or node
arising from noise produced by another component or node. They contain the :class:`noise source <.components.Noise>`
that produces the noise and a reference to the component or node that the noise is measured at, and
its units. :class:`Multi-noise spectra <.data.MultiNoiseDensity>` contain a list of multiple noise
sources; these are used to represent noise sums.
