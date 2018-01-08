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
from .liso import InputParser

PROG = "circuit"
SYNOPSIS = "{} <command> [<args>...]".format(PROGRAM)

# NOTE: double spaces are interpreted by text2man to be paragraph
# breaks.  NO DOUBLE SPACES.  Also two spaces at the end of a line
# indicate an element in a tag list.
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

class Cmd(object):
    """Base class for commands"""

    cmd = ""

    def __init__(self):
        """Initialise argument parser"""

        self.parser = argparse.ArgumentParser(
            prog="{} {}".format(PROG, self.cmd),
            description=self.__doc__.strip()
        )

    def parse_args(self, args):
        """Parse arguments and return :class:`argparse.Namespace` object

        :param args: arguments
        """

        return self.parser.parse_args(args)

    def __call__(self, args):
        """Take Namespace object as input and execute command"""

        pass

class Liso(Cmd, metaclass=abc.ABCMeta):
    """LISO operations"""

    cmd = "liso"

    def __init__(self):
        super(Liso, self).__init__()

        self.parser.add_argument("input_file", help="LISO input file")
        self.parser.add_argument("--print-equations", action="store_true",
                                 help="print circuit equations")
        self.parser.add_argument("--print-matrix", action="store_true",
                                 help="print circuit matrix")
        self.parser.add_argument("-v", "--verbose", action="store_true",
                                 help="enable verbose output")

    def __call__(self, args):
        if args.verbose:
            logging_on()

        parser = InputParser(args.input_file)
        parser.show(print_equations=args.print_equations,
                    print_matrix=args.print_matrix, print_progress=args.verbose)

class Help(Cmd):
    """Print manpage or command help (also '-h' after command)."""

    cmd = "help"

    def __init__(self):
        Cmd.__init__(self)
        self.parser.add_argument("cmd", nargs="?",
                                 help="command")

    def __call__(self, args):
        if args.cmd:
            get_func(args.cmd).parser.print_help()
        else:
            print(MANPAGE.format(cmds=format_commands(man=True)))

CMDS = collections.OrderedDict([
    ("liso", Liso),
    ("help", Help),
    ])

ALIAS = {
    "--help": "help",
    "-h": "help",
    }

##################################################

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
        # no command specified; print error message
        print("Command not specified.", file=sys.stderr)
        print("usage: " + SYNOPSIS, file=sys.stderr)
        print(file=sys.stderr)

        # print commands
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
