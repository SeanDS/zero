# circuit.py
Linear electronic circuit simulator utility. This package provides tools to
simulate transfer functions and noise in linear electronic circuits, SI unit
parsing and formatting and more.

This tool is inspired by, and partially based on, [LISO](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso)
and [Elektrotickle](https://github.com/tobin/Elektrotickle/). It also (somewhat)
understands LISO input and output files.

## Installation
Installation is best handled using `pip`. As the library is Python 3 only, on
some systems you must install this using `pip3` instead:
```bash
pip3 install git+https://git.ligo.org/sean-leavey/circuit.git
```
This installs the library and adds a console script `circuit` which provides
access to the package's command line utility.

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

## Basic usage
There is a very basic CLI provided by the program. Open up a terminal and type:
```bash
circuit help
```
for a list of available commands. Run `circuit help command` for more detailed
help for a particlar `command`.

For examples of how to build simulation scripts with Python, see the `examples`
directory.

## Current limitations

### Solver
  - No pre-scaling is performed on matrices and as such they can contain values
    that differ by tens of orders of magnitude, potentially leading to numerical
    rounding issues in extreme cases.

### LISO parsing
  - Some LISO commands not yet supported:
    - all root mode commands
    - no fit commands
    - m
    - factor
    - maxinput
    - zin
    - opdiff
    - opstab
    - noisy
    - gnuterm
    - inputnoise
  - `noise` command's plot options are ignored (all noise sources are plotted
    including incoherent sum)
  - Coordinates in LISO files (e.g. `im`, `deg+`, etc.) are ignored in favour of
    `db` and `deg` in all cases
  - Output parser assumes all outputs are in dB and degrees (noise columns are
    handled appropriately, however)
  - LISO's op-amp library format is not supported (see `Op-amp library` below)

### Op-amp library
The op-amp library is implemented in a different format to that of LISO,
primarily for logistical reasons: Python contains a convenient `ConfigParser`
library which can read and write config files similar to Windows `INI` files,
but in a slightly different format to LISO's op-amp library format. The main
difference is that in `ConfigParser` files, repeated terms are not allowed in
the same entry, so LISO's use of multiple "pole" or "zero" entries under an
op-amp are not supported. Instead, the library represents poles and zeros as
single line expressions of comma separated values:
```
[op177]
...
poles = 7.53M 1.78, 1.66M # fitted from measurement
...
```
Furthermore, the library improves on that of LISO's by allowing an
"alias" setting where you can specify other op-amps with the same properties:
```
[tl074]
aliases = tl084
...
```

A LISO op-amp library parser will be added at a later date.

## Future ideas
  - Allow arbitrary op-amp noise spectra (interpolate to the frequency vector
    actually used)
  - Some sort of system for sharing op-amp, regulator, resistor, etc. library
    data across the web
  - A standardised export file format (XML?)
  - Other types of noise, e.g. resistor excess noise
  - SciPy/Matlab system object export?
  - Visualise circuit node network with graphviz
  - Op-amp noise optimisation: here's my circuit, this is the frequency band I
    care about, now what's the best op-amp to use?
  - Multiple voltage/current inputs?
  - Grouped components that are represented as a single component in the input
    definition:
      - filters, e.g. whitening filters
      - real passive components: capacitors with ESR, resistors with stray
        inductance, etc.
  - Parallelised solving (need to be careful about thread safety)

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  

Invaluable insight into LISO's workings provided by Gerhard Heinzel.
