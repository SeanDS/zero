# Datasheet grabber
View datasheets using the command line:
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

If there are multiple datasheets available, a list sorted by date
is shown so you can select which to download. To force the utility
to show only the first result, use the `--first` or `-f` flag:
```bash
datasheet --first AD797
```

By default, wildcard characters are added to the search string. To disable
this and instead search for an exact part name, use the `--exact` or `-e` flag:
```bash
datasheet --exact TL074CN
```

By default, the datasheet will be downloaded to a temporary location. To specify
a directory to save the PDF in, use the `--path` or `-p` flag:
```bash
datasheet --path datasheets OP27
```

To download but not display the datasheet, specify `--download-only` or `-d`. This
is useful in combination with `--path` to archive datasheets:
```bash
datasheet --path datasheets --download-only OP27
```

Verbose output can be enabled with `--verbose` or `-v`.

## Contributing
Bug reports and feature requests are always welcome, as are contributions to 
the code. Please use the project's [issue tracker](https://git.ligo.org/sean-leavey/datasheet/issues).

## Credits
Sean Leavey  
<sean.leavey@ligo.org>  