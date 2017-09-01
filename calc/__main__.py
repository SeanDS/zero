#!/usr/bin/env python3

import os
import io
import sys
import logging
import argparse
import textwrap
import collections

from .regulator import Regulator

PROG = 'calc'
DESC = 'Electronics calculator'

SYNOPSIS = '{} <command> [<args>...]'.format(PROG)

# NOTE: double spaces are interpreted by text2man to be paragraph
# breaks.  NO DOUBLE SPACES.  Also two spaces at the end of a line
# indicate an element in a tag list.
MANPAGE = """
NAME
  {prog} - {desc}

SYNOPSIS
  {synopsis}

DESCRIPTION

  Electronics calculator utility. Provides tools to calculate resistor values
  for regulator circuits.

COMMANDS

{{cmds}}

AUTHOR
    Sean Leavey <electronics@attackllama.com>
""".format(prog=PROG,
           desc=DESC,
           synopsis=SYNOPSIS,
           ).strip()

def enable_verbose_logs():
    # set up logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout)

class Cmd(object):
    """base class for commands"""

    cmd = ''

    def __init__(self):
        """Initialize argument parser"""
        self.parser = argparse.ArgumentParser(
            prog='{} {}'.format(PROG, self.cmd),
            description=self.__doc__.strip(),
            # formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    def parse_args(self, args):
        """Parse arguments and returned ArgumentParser Namespace object"""

        return self.parser.parse_args(args)
    def __call__(self, args):
        """Take Namespace object as input and execute command"""

        pass

class RegulatorResistors(Cmd):
    """Calculate regulator resistor values"""

    cmd = 'regres'

    def __init__(self):
        Cmd.__init__(self)
        self.parser.add_argument('voltage',
                                 help="target voltage")
        self.parser.add_argument('type',
                                 help="regulator type")
        self.parser.add_argument('-n', '--num-values', default=3,
                                 help="number of closest values to show")
        self.parser.add_argument('-s', '--series',
                                 help="resistor series")
        self.parser.add_argument('-v', '--verbose', action='store_true',
                                 help="enable verbose output")
        # self.parser.add_argument('field', nargs='?',
        #                          help="document field")

    def __call__(self, args):
        if args.verbose:
            enable_verbose_logs()

        voltage = float(args.voltage)
        reg = Regulator(args.type)
        num_values = int(args.num_values)

        matches = reg.resistors_for_voltage(voltage, n_values=num_values,
                                            series=args.series)

        print('Closest {} matches for {}V target:'.format(num_values,
                                                          args.voltage))
        for match in matches:
            # how close, in percent, the match is to the desired voltage
            closeness = 100 * (1 - match[0] / voltage)

            # formatted string
            closeness_str = "{:.2f}".format(abs(closeness))

            if closeness > 0:
                closeness_sign = "-"
            else:
                closeness_sign = "+"

            print('V = {} ({}{}%), R1 = {}, R2 = {}'.format(match[0],
                closeness_sign, closeness_str, match[1], match[2]))

class Help(Cmd):
    """Print manpage or command help (also '-h' after command)."""

    cmd = 'help'

    def __init__(self):
        Cmd.__init__(self)
        self.parser.add_argument('cmd', nargs='?',
                                 help="command")

    def __call__(self, args):
        if args.cmd:
            get_func(args.cmd).parser.print_help()
        else:
            print(MANPAGE.format(cmds=format_commands(man=True)))

CMDS = collections.OrderedDict([
    ('regres', RegulatorResistors),
    ('help', Help),
    ])

ALIAS = {
    '--help': 'help',
    '-h': 'help',
    }

##################################################

def format_commands(man=False):
    prefix = ' '*8
    wrapper = textwrap.TextWrapper(
        width=70,
        initial_indent=prefix,
        subsequent_indent=prefix,
        )

    with io.StringIO() as f:
        for name, func in CMDS.items():
            if man:
                fo = func()
                usage = fo.parser.format_usage()[len('usage: {} '.format(PROG)):].strip()
                desc = wrapper.fill('\n'.join([l.strip() for l in fo.parser.description.splitlines() if l]))
                f.write('  {}  \n'.format(usage))
                f.write(desc+'\n')
                f.write('\n')
            else:
                desc = func.__doc__.splitlines()[0]
                f.write('  {:10}{}\n'.format(name, desc))
        output = f.getvalue()

    return output.rstrip()

def get_func(cmd):
    if cmd in ALIAS:
        cmd = ALIAS[cmd]
    try:
        return CMDS[cmd]()
    except KeyError:
        print('Unknown command:', cmd, file=sys.stderr)
        print("See 'help' for usage.", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print('Command not specified.', file=sys.stderr)
        print('usage: '+SYNOPSIS, file=sys.stderr)
        print(file=sys.stderr)
        print(format_commands(), file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]
    func = get_func(cmd)
    func(func.parse_args(args))

##################################################

if __name__ == '__main__':
    main()
