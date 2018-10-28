# Zero
Linear electronic circuit simulator utility. This package provides tools to
simulate transfer functions and noise in linear electronic circuits, SI unit
parsing and formatting and more.

This tool is inspired by [LISO](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso).
It also (somewhat) understands LISO input and output files, and can plot or
re-simulate their contents.

## Documentation
See the [online documentation](https://docs.ligo.org/sean-leavey/zero/latest/).

## Installation
This library requires that Python 3 is installed. It has been tested on 3.5, 3.6 and 3.7,
but may work on earlier versions of Python 3. Python 2 is not supported. You may wish to use
`virtualenv` or `conda` to manage a separate environment with Python 3.

This library contains a `setup.py` file which tells Python how it should be
installed. Installation can be automated using `pip`. Open up a terminal or
command prompt (Windows) and type:
```bash
pip install git+https://git.ligo.org/sean-leavey/zero.git
```
This installs the library and adds a console script `zero` which provides
access to the package's command line utility.

If you want to update the library to a later version after having previously
installed it, run:
```bash
pip install git+https://git.ligo.org/sean-leavey/zero.git --upgrade
```

## Contributing
Bug reports and feature requests are always welcome, as are code contributions. Please use the
project's [issue tracker](https://git.ligo.org/sean-leavey/zero/issues).

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

The author is grateful for additional contributions by Sebastian Steinlechner.
