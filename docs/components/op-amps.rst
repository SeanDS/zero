.. include:: /defs.txt

.. currentmodule:: zero.components

Op-amps
-------

:class:`Op-amps <OpAmp>` in |Zero| take differential inputs and provide a
single output.

Voltage gain
============

The :meth:`voltage gain <OpAmp.gain>` of an op-amp is defined by its open
loop gain (`a0`), gain-bandwidth product (`gbw`), delay and poles or
zeros. The gain is as function of frequency.

Special case: voltage followers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an op-amp is configured as a `voltage follower` (otherwise known as a
`buffer`), where the output node is the same as the inverting input node,
the voltage gain is modified.

Noise
=====

Op-amps produce :class:`voltage noise <VoltageNoise>` across their input
and output :class:`nodes <Node>`, and :class:`current noise <CurrentNoise>`
is present at its input :class:`nodes <Node>`. The noise is a function of
frequency.

Library
=======
