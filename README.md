# circuit.py
Linear electronic circuit simulator utility. This package provides tools to
simulate transfer functions and noise in linear electronic circuits, SI unit
parsing and formatting and more.

This tool is inspired by [LISO](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso).
It also (somewhat) understands LISO input and output files, and can plot or
re-simulate their contents.

## Program and library
The simulator tries to replicate LISO's operation: a small signal ac analysis.
The electrical components in a circuit are defined using their nodal
connections, and the program solves a matrix equation to obtain either transfer
functions between nodes/components or noise at a particular node.

Under the hood, components in the circuit generate their own equations in the
circuit matrix, using Kirchoff's voltage and current laws as described in the
[LISO manual](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso#manual).

Credit goes to Tobin Fricke's [Elektrotickle](https://github.com/tobin/Elektrotickle/)
for providing easy to read source code showing an approach to solving a circuit
problem. This library evolves Elektrotickle to use a more object-oriented
structure, support for current transfer functions, much more comprehensive plotting,
tools to make direct comparisons to LISO, displaying of circuit equations and more.

## Installation
This library requires that Python 3 is installed. It has been tested on version 3.5, but
may work on earlier versions of Python 3. Python 2 is not supported.

If you don't have Python 3 installed, have a look at [this](https://www.python.org/downloads/).

This library contains a `setup.py` file which tells Python how it should be
installed. Installation can be automated using `pip`. Open up a terminal or
command prompt (Windows) and type:
```bash
pip install git+https://git.ligo.org/sean-leavey/circuit.git
```
This installs the library and adds a console script `circuit` which provides
access to the package's command line utility.

If you want to update the library to a later version after having previously
installed it, run:
```bash
pip install git+https://git.ligo.org/sean-leavey/circuit.git --upgrade
```

## Basic usage
There is a very basic CLI provided by the program. Open up a terminal and type:
```bash
circuit --help
```
for a list of available commands. Run `circuit command --help` for more detailed
help for a particular `command`.

### Run LISO files
```bash
circuit liso path/to/liso/file
```

`circuit.py` can parse both LISO input (`.fil`) and LISO output (`.out`) files.
The above command will display the results. Some commands are not yet supported
(see `LISO parsing` below).

#### Parser hints
To force a file to be parsed as either an input or an output file, specify the
`--force-input` or `--force-output` flags.

### Comparing results to LISO
A comparison between `circuit.py`'s native result and that of LISO can be made
with `circuit liso-compare path/to/liso/file`. Note that any operations that
involve running LISO (e.g. `liso-compare`) require the LISO binary to be set
using the `LISO_DIR` environment variable.

LISO can also be run by `circuit.py` directly, using the `circuit liso-external`
command. To allow LISO to plot its own results, instead of plotting the results
in `circuit.py`, specify the `--liso-plot` flag.

### As a library
`circuit.py` can also be included as a library within other Python code. For
examples of how to build simulation scripts with Python, see the `examples`
directory.

## Tests
The script in `/tests/runner.py` can be run to automatically test `circuit.py`.
There are various tests which compare the results of simulations to LISO; these
can be run with `runner.py validation`. To run all tests, call `runner.py` with
the `all` argument.

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
  - Other analyses, e.g. DC operating point (linearises circuits for AC analyses,
    thereby allowing nonlinear components like diodes to be modelled)

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  

Invaluable insight into LISO's workings provided by Gerhard Heinzel.

The author is grateful for additional contributions by:
  - Sebastian Steinlechner
