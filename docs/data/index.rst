.. include:: /defs.txt

.. currentmodule:: zero.data

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

The response's underlying complex data is stored in its :attr:`~.Response.complex_magnitude`
property. The magnitude and phase can be retrieved using the :attr:`~.Response.magnitude` and
:attr:`~.Response.phase` properties, respectively.

.. note::

   The :attr:`~.Response.magnitude` is returned with decibel (power) scaling, i.e. :math:`20 \log_{10} \left| x \right|`
   where :math:`x` is the complex response. The :attr:`~.Response.phase` is returned in units of
   (unwrapped) degrees.

.. code-block:: python

   >>> response.complex_magnitude
   array([-1.44905660e+06+271698.11320755j, -1.28956730e+06+520929.0994604j ,
          -8.53524671e+05+742820.7338082j , -3.32179931e+05+622837.37024221j,
          -8.66146537e+04+349885.52751013j, -1.95460509e+04+170108.87173014j,
          -4.25456479e+03 +79773.08987768j, -9.18662496e+02 +37109.9690498j ,
          -1.98014980e+02 +17233.2022651j , -4.26654531e+01  +7999.77245092j])
   >>> response.magnitude
   array([123.37176609, 122.86535272, 121.07307338, 116.97464649,
          111.13682633, 104.67150284,  98.04946401,  91.39247246,
          84.72789306,  78.06167621])
   >>> response.phase
   array([169.38034472, 158.00343425, 138.96701713, 118.07248694,
          103.90412599,  96.55472157,  93.05288251,  91.41807544,
          90.65831778,  90.30557459])

Noise spectral densities
------------------------

:class:`Noise spectral densities <.data.NoiseDensity>` contain the noise at a particular component
or node arising from noise produced by another component or node. They contain the :class:`noise source <.components.Noise>`
that produces the noise and a reference to the component or node that the noise is measured at, and
its units. :class:`Multi-noise spectra <.data.MultiNoiseDensity>` contain a list of multiple noise
sources; these are used to represent noise sums.

The noise spectral density's underlying data is stored in its :attr:`~.NoiseDensityBase.spectral_density`
property.

.. code-block:: python

   >>> response.spectral_density
   array([1.29259971e-07, 1.00870891e-07, 8.45132667e-08, 7.57294937e-08,
          7.12855936e-08, 6.91259094e-08, 6.81002020e-08, 6.76188164e-08,
          6.73941734e-08, 6.72894850e-08])
