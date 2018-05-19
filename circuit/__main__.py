#!/usr/bin/env python3

"""Circuit simulator utility"""

import io
import sys
import abc
import logging
import argparse
import textwrap
import collections

from circuit import __version__, DESCRIPTION, PROGRAM, logging_on
from .liso import InputParser, OutputParser, InvalidLisoFileException

PROG = "circuit"
AUTHOR = "Sean Leavey <electronics@attackllama.com>"
SYNOPSIS = "{} <command> [<args>...]".format(PROGRAM)
HEADER = """{prog} {version}
{author}
""".format(prog=PROGRAM, version=__version__, author=AUTHOR)
MANPAGE = """
NAME
  {prog} {version}

SYNOPSIS
  {synopsis}

DESCRIPTION

  {desc}

COMMANDS

{{cmds}}

AUTHOR
    Sean Leavey <sean.leavey@ligo.org>
""".format(prog=PROGRAM, version=__version__, desc=DESCRIPTION,
           synopsis=SYNOPSIS).strip()

LOGGER = logging.getLogger()

class Cmd(object, metaclass=abc.ABCMeta):
    """Base class for commands"""

    cmd = ""

    def __init__(self):
        """Initialise argument parser"""

        self.parser = argparse.ArgumentParser(
            prog="{} {}".format(PROG, self.cmd),
            description=self.__doc__.strip()
        )

        self.parser.add_argument("-v", "--verbose", action="store_true",
                                 help="enable verbose output")

    def parse_args(self, args):
        """Parse arguments and return :class:`argparse.Namespace` object

        :param args: arguments
        """

        return self.parser.parse_args(args)

    def __call__(self, args):
        """Execute command within a try/except block, obeying verbosity"""
        if args.verbose:
            logging_on()

            # print title and version
            print(HEADER)

        try:
            self.call(args)
        except Exception as e:
            print("Error: %s" % e, file=sys.stderr)
            raise e

    @abc.abstractmethod
    def call(self, args):
        """Take Namespace object as input and execute command"""
        return NotImplemented

class Sim(Cmd):
    """Parse either a LISO input file or output file, simulate its circuit,
    then plot the results. The existing data in LISO output files is ignored.

    To directly plot the results of a LISO output file, see \"liso\"."""

    cmd = "sim"

    def __init__(self):
        super(Sim, self).__init__()

        self.parser.add_argument("file", help="path to LISO input or output file")
        self.parser.add_argument("--print-equations", action="store_true",
                                 help="print circuit equations")
        self.parser.add_argument("--print-matrix", action="store_true",
                                 help="print circuit matrix")

    def call(self, args):
        # try parsing first as an input file, then an output file
        try:
            parser = InputParser(args.file)
            LOGGER.debug("parsed as LISO input file")

            parser.show(print_equations=args.print_equations,
                        print_matrix=args.print_matrix,
                        print_progress=args.verbose)
        except InvalidLisoFileException:
            LOGGER.debug("attempt to parse file as LISO input failed, trying "
                         "to parse as output instead")
            # try as output file
            parser = OutputParser(args.file)
            LOGGER.debug("parsed as LISO output file")
            solution = parser.run_native(print_equations=args.print_equations,
                                         print_matrix=args.print_matrix,
                                         print_progress=args.verbose)
            solution.plot()

class Liso(Cmd):
    """Plot a LISO output file.

    To model a LISO output file using this utility's built-in simulator, see
    \"sim\".
    """

    cmd = "liso"

    def __init__(self):
        super(Liso, self).__init__()

        self.parser.add_argument("output_file", help="LISO output file")

    def call(self, args):
        if args.verbose:
            logging_on()

        parser = OutputParser(args.output_file)
        parser.show()

class Help(Cmd):
    """Print manpage or command help (also "-h" after command)."""

    cmd = "help"

    def __init__(self):
        Cmd.__init__(self)
        self.parser.add_argument("cmd", nargs="?",
                                 help="command")

    def call(self, args):
        if args.cmd:
            get_func(args.cmd).parser.print_help()
        else:
            print(MANPAGE.format(cmds=format_commands(man=True)))

CMDS = collections.OrderedDict([
    ("sim", Sim),
    ("liso", Liso),
    ("help", Help),
])

ALIAS = {
    "--help": "help",
    "-h": "help",
}

def format_commands(man=False):
    """Generate documentation for available commands"""

    # documentation indentation
    prefix = " " * 8

    # documentation text format
    wrapper = textwrap.TextWrapper(
        width=70,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )

    with io.StringIO() as stream:
        for name, func in CMDS.items():
            if man:
                command = func()

                # format usage
                usage = command.parser.format_usage()[len("usage: {} ".format(PROG)):].strip()

                # format description
                desc = wrapper.fill("\n".join([line.strip()
                                               for line in command.parser.description.splitlines()
                                               if line]))

                # print documentation
                stream.write("  {}  \n".format(usage))
                stream.write(desc + "\n")
                stream.write("\n")
            else:
                desc = func.__doc__.splitlines()[0]
                stream.write("  {:10}{}\n".format(name, desc))

        output = stream.getvalue()

    return output.rstrip()

def get_func(cmd):
    """Find command from specified string

    :param cmd: command string
    """

    if cmd in ALIAS:
        # get the command the alias points to
        cmd = ALIAS[cmd]

    try:
        # return command if it exists
        return CMDS[cmd]()
    except KeyError:
        # command not found; print error message
        print("Unknown command:", cmd, file=sys.stderr)
        print("See 'help' for usage.", file=sys.stderr)

        # exit with error code
        sys.exit(1)

def main():
    """Main program"""

    if len(sys.argv) < 2:
        # no command specified

        # print title and version
        print(HEADER)

        # print usage
        print("Command not specified.", file=sys.stderr)
        print("usage: " + SYNOPSIS, file=sys.stderr)
        print(file=sys.stderr)

        # print available commands
        print(format_commands(), file=sys.stderr)

        # exit with error code
        sys.exit(1)

    # parse command
    cmd = sys.argv[1]

    # parse arguments
    args = sys.argv[2:]

    # get and execute command
    func = get_func(cmd)
    func(func.parse_args(args))

if __name__ == "__main__":
    main()
