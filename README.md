# circuit.py
Linear electronic circuit simulator utility. This package provides tools to
simulate transfer functions and noise in linear electronic circuits, SI unit
parsing and formatting and more.

This tool is inspired by, and partially based on, [LISO](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso)
and [Elektrotickle](https://github.com/tobin/Elektrotickle/). It also (somewhat)
understands LISO input and output files.

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

## Installation
This library requires that Python 3 is installed. It has been tested on version
3.5, but it might worth on 3.4. If you don't have Python 3 installed, have a
look at [this](https://www.python.org/downloads/).

This library contains a `setup.py` file which tells Python how it should be
installed. Installation can be automated using `pip`, or, on some systems,
`pip3`. Open up a terminal (Linux, Mac, etc.) or command prompt (Windows) and
type:
```bash
pip3 install git+https://git.ligo.org/sean-leavey/circuit.git
```
This installs the library and adds a console script `circuit` which provides
access to the package's command line utility.

If you want to update the library to a later version after having previously
installed it, run:
```bash
pip3 install git+https://git.ligo.org/sean-leavey/circuit.git --upgrade
```

## Basic usage
There is a very basic CLI provided by the program. Open up a terminal and type:
```bash
circuit help
```
for a list of available commands. Run `circuit help command` for more detailed
help for a particular `command`.

`circuit.py` can also be included as a library within other Python code. For
examples of how to build simulation scripts with Python, see the `examples`
directory.

## Tests
The script in `/tests/liso/liso.py` can be run to automatically test the
solver against LISO with a set of LISO input files. Currently, most scripts
produce results that agree with LISO outputs to 1 part in 10,000 (both
relative and absolute). Certain circuits with very small numbers (for example,
transfer functions around -100 dB or more) do not always agree within this
bound, possibly due to numerical precision of the solver routine.

## Current limitations

### LISO parsing
  - Coordinates for output signals (e.g. `im`, `deg+`, etc.) are ignored in
    favour of `db` and `deg` in all cases
  - Output parser assumes all outputs are in dB and degrees (noise columns are
    handled appropriately, however)
  - LISO's op-amp library format is not supported, but the full library bundled
    with LISO is implemented in a different format (see `Op-amp library` below)
  - Some LISO commands not yet supported. Here are some that might be supported
    in the future, in rough order of priority (first highest):
    - `factor` (input multiplicative factor)
    - `m` (mutual inductance)
    - `noisy` (switch on/off noise from specific components)
    - `inputnoise` (circuit noise referred to input node)
    - `zin` (input impedance)
    - `opdiff` (plot op-amp input differential voltage)
    - `margin` (compute op-amp phase margin; replaces `opstab` in LISO v1.78)
    - `sens` (print table of component sensitivities)
  - And here are some commands which will probably not be implemented:
    - commands associated with root mode and fitting (tools such as `vectfit`
      may be suitable replacements)
    - other `max` or `min` based commands, e.g. `maxinput` (need fitting?)
    - `eagle` (produce EAGLE file)    
    - `gnuterm`
    - component `C0805` (0805 capacitor with parasitic properties; not
      implemented in favour of grouped components feature idea below)
  - `noise` command's plot options are ignored (all noise sources are plotted
    including incoherent sum)

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
Finally, the English convention of using "v" to represent voltage instead of "u"
has been used, so `un` and `uc` are instead `vn` and `vc`.

A LISO op-amp library parser may be added at a later date.

### Tests
  - Comparisons of complex value series don't handle phase wrapping, and so
    occasionally flag two matched series as different. Might be better to
    compare complex numbers instead.

## Contributing
Bug reports and feature requests are always welcome, as are contributions to the
code. Please use the project's [issue tracker](https://git.ligo.org/sean-leavey/circuit/issues).

## Future ideas
  - Return plot objects to allow user to modify them
  - Allow arbitrary op-amp noise spectra (interpolate to the frequency vector
    actually used)
  - Split op-amp families into their own library files
  - Some sort of system for sharing op-amp, regulator, resistor, etc. library
    data across the web
  - Breakout data classes into separate project (TFs, noise and data handling
    are probably useful for other purposes)
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
  - Warn user if numerical precision might prevent LISO agreement (e.g. for
    magnitudes <100 dB)

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  

Invaluable insight into LISO's workings provided by Gerhard Heinzel.

The author is grateful for additional contributions by:
  - Sebastian Steinlechner
