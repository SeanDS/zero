.. include:: /defs.txt

.. currentmodule:: zero.data

###############
Data containers
###############

.. code-block:: python

   >>> from zero.data import Series, Response, NoiseDensity

|Zero| :ref:`analysis <analyses/index:Analyses>` results (responses and noise spectra) are stored
within `function` containers. These are relatively low level objects that hold each function's data
(within a :ref:`series <data/index:Series>`), its frequency axis, and any meta data produced by the
analysis. These objects are able to plot themselves when provided a figure to draw to. They also
contain logic to compare themselves to other functions, to check for equivalency.

In normal circumstances, you should not need to directly interact with these objects; rather, you
can plot and save their underlying data using a :ref:`Solution <solution/index:Solutions>`.

Series
------

Underlying function data is stored in a :class:`.Series`. This contains two dimensional data. Series
support basic mathematical operations such as multiplication, division and inversion.

Functions
---------

Responses
~~~~~~~~~

:class:`Responses <.data.Response>` contain the response of a component or node to another component
or node. Each response contains references to the source and sink component or node, and its units.

The response's underlying complex data is stored in its :attr:`~.Response.complex_magnitude`
property. The magnitude and phase can be retrieved using the :attr:`~.Response.magnitude` and
:attr:`~.Response.phase` properties, respectively. The decibel-scaled magnitude can be retrieved
using :attr:`~.Response.db_magnitude`.

.. note::

   :attr:`~.Response.db_magnitude` is returned with power scaling, i.e.
   :math:`20 \log_{10} \left| x \right|` where :math:`x` is the complex response.

.. code-block:: python

   >>> response.complex_magnitude
   array([-1.44905660e+06+271698.11320755j, -1.28956730e+06+520929.0994604j ,
          -8.53524671e+05+742820.7338082j , -3.32179931e+05+622837.37024221j,
          -8.66146537e+04+349885.52751013j, -1.95460509e+04+170108.87173014j,
          -4.25456479e+03 +79773.08987768j, -9.18662496e+02 +37109.9690498j ,
          -1.98014980e+02 +17233.2022651j , -4.26654531e+01  +7999.77245092j])
   >>> response.db_magnitude
   array([123.37176609, 122.86535272, 121.07307338, 116.97464649,
          111.13682633, 104.67150284,  98.04946401,  91.39247246,
          84.72789306,  78.06167621])
   >>> response.phase
   array([169.38034472, 158.00343425, 138.96701713, 118.07248694,
          103.90412599,  96.55472157,  93.05288251,  91.41807544,
          90.65831778,  90.30557459])

Noise spectral densities
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`Noise spectral densities <.data.NoiseDensity>` contain the noise at a particular component
or node arising from noise produced by another component or node. They contain the :class:`noise
source <.components.Noise>` that produces the noise and a reference to the component or node that
the noise is measured at, and its units. :class:`Multi-noise spectra <.data.MultiNoiseDensity>`
contain a list of multiple noise sources; these are used to represent noise sums.

The noise spectral density's underlying data is stored in its
:attr:`~.NoiseDensityBase.spectral_density` property.

.. code-block:: python

   >>> response.spectral_density
   array([1.29259971e-07, 1.00870891e-07, 8.45132667e-08, 7.57294937e-08,
          7.12855936e-08, 6.91259094e-08, 6.81002020e-08, 6.76188164e-08,
          6.73941734e-08, 6.72894850e-08])

Labels
~~~~~~

Functions can have labels that are used in plot legends and when :ref:`searching for functions in a
solution <solution/index:Retrieving functions>`.

Labels can be set for functions using their :attr:`~.data.BaseFunction.label` property. If no label
is set by the user, a default label is produced using the function's source and sink in the case of
single-source and -sink functions, or "Incoherent sum" for :class:`noise sums <.MultiNoiseDensity>`.

Mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~

The underlying data within a function can be multiplied, divided and inverted by applying
mathematical operations to the function object. Multiplication and division can be applied using
scalars or other functions. For example, :ref:`noise spectra <data/index:Noise spectral densities>`
can be multiplied by :ref:`responses <data/index:Responses>` to project noise to a different part of
a circuit (used for example to :ref:`refer noise to the circuit input <analyses/ac/noise:Referring
noise to the input>`).

When an operation involves two functions, the units of each function are checked for
validity. As determined by the order of operation, the left function's sink must have the same units
as the right function's source. The resulting function then takes the left functions' source and the
right function's sink.

.. hint::

    While the inner sources and sinks of such operations must have the same units, they do not need
    to be the same :class:`element <.BaseElement>`. This is to allow functions to be lightweight and
    not have to maintain a reference to the component, node or noise source objects they originally
    represented (rather, just their label). It is up to the user to check that each operation makes
    physical sense.

Some operations are not possible, such as multiplying noise by noise. In these cases, a
:class:`ValueError` is raised.
