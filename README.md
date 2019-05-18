# Zero
Linear electronic circuit utility. This package provides tools to simulate responses and noise in
linear electronic circuits, SI unit parsing and formatting, datasheet grabbing, and more.

This tool's simulator is inspired by [LISO](https://wiki.projekt.uni-hannover.de/aei-geo-q/start/software/liso),
and comes bundled with its op-amp library ([including tools to search it](https://docs.ligo.org/sean-leavey/zero/cli/library.html#search-queries)).
It also ([somewhat](https://docs.ligo.org/sean-leavey/zero/liso/input.html#known-incompatibilities))
understands LISO input and output files, and can plot or re-simulate their contents.

## Documentation
See the [online documentation](https://docs.ligo.org/sean-leavey/zero/).

## Installation
This library requires at least Python 3.6. It will not work on earlier versions of Python 3, nor
Python 2. You may wish to use `virtualenv` or `conda` to manage a separate environment with Python
3.

This library contains a `setup.py` file which tells Python how it should be installed. Installation
can be automated using `pip`. Open up a terminal or command prompt (Windows) and type:
```bash
pip install zero
```
This installs the library and adds a console script `zero` which provides access to the package's
command line utility.

If you want to update the library to a later version after having previously installed it, run:
```bash
pip install zero --upgrade
```

## Contributing
Bug reports and feature requests are always welcome, as are code contributions. Please use the
project's [issue tracker](https://git.ligo.org/sean-leavey/zero/issues).

## Future ideas
  - Allow arbitrary op-amp noise spectra (interpolate to the frequency vector actually used)
  - Split op-amp families into their own library files
  - Some sort of system for sharing op-amp, regulator, resistor, etc. library data across the web
  - A standardised export file format (XML?)
  - Other types of noise, e.g. resistor excess noise
  - Op-amp noise optimisation: here's my circuit, this is the frequency band I care about, now
    what's the best op-amp to use?
  - Grouped components that are represented as a single component in the input definition:
      - filters, e.g. whitening filters
      - real passive components: capacitors with ESR, resistors with stray inductance, etc.

## Credits
Sean Leavey  
<sean.leavey@ligo.org>

Invaluable insight into LISO's workings provided by Gerhard Heinzel. The author is also grateful for
contributions by Sebastian Steinlechner.
