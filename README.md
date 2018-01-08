# circuit.py
Linear electronic circuit simulator utility. This package provides tools to
simulate transfer functions and noise in linear electronic circuits, SI unit
parsing and formatting and more.

## Installation
Installation is best handled using `pip`. As the library is Python 3 only, on
some systems you must install this using `pip3` instead:
```bash
pip3 install git+https://git.ligo.org/sean-leavey/circuit.git
```

## Program and library
The simulator tries to replicate LISO's operation: a small signal ac analysis.
The electrical components in a circuit are defined using their nodal
connections, and the program solves a matrix equation to obtain either transfer
functions between nodes/components or noise at a particular node.

Under the hood, components in the circuit generate their own equations in the
circuit matrix, using Kirchoff's voltage and current laws as described in the
[LISO manual](http://www2.mpq.mpg.de/~ros/geo600_docu/soft/liso/manual.pdf).

Credit goes to Tobin Fricke's [Elektrotickle](https://github.com/tobin/Elektrotickle/)
for providing easy to read source code showing an approach to solving a circuit
problem. This library evolves Elektrotickle to use a more object-oriented
structure, support for current transfer functions, direct comparison to LISO
results and more advanced plotting, but the basic solving functionality is
almost the same.

## Future ideas
  - Allow arbitrary op-amp noise spectra (interpolate to the frequency vector
    actually used)
  - Some sort of system for sharing op-amp, regulator, resistor, etc. library
    data across the web
  - An standardised export file format (XML?)
  - Other types of noise, e.g. resistor excess noise
  - SciPy/Matlab system object export?
  - Visualise circuit node network with graphviz
  - Op-amp noise optimisation: here's my circuit, this is the frequency band I
    care about, now what's the best op-amp to use?
  - Multiple voltage/current inputs?
  - Grouped components, like whitening filters, that are represented as a single
    component in the input definition

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  
https://github.com/SeanDS

Invaluable insight into LISO's workings provided by Gerhard Heinzel.
