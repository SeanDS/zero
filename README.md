# Datasheet grabber
View datasheets from the command line:
```bash
datasheet AD829
```

Datasheets are searched for using [Octopart](https://octopart.com/)'s API,
downloaded and shown using your default PDF viewer.

## Installation
There is a `setup.py` file which can be used to install the program on
your system path. Alternatively the installation can be managed with
`pip`:
```bash
pip install --user git+https://git.ligo.org/sean-leavey/datasheet
```

## Usage
View the available commands with
```bash
datasheet --help
```

To download and view a datasheet, type e.g.:
```bash
datasheet AD797
```

By default, wildcard characters are added to the search string. To disable
this and instead search for an exact part name, use the `--exact` flag:
```bash
datasheet --exact TL074
```

## Contributing
Bug reports and feature requests are always welcome, as are contributions to 
the code. Please use the project's [issue tracker](https://git.ligo.org/sean-leavey/datasheet/issues).

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  