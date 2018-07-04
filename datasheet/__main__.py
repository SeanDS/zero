#!/usr/bin/env python

"""Datasheet utility"""

import sys
import logging
import argparse

from . import __version__, PROGRAM, DESCRIPTION, logging_on
from .request import DatasheetRequest

LOGGER = logging.getLogger(__name__)

class Parser(object):
    def __init__(self, program, version, err_stream=sys.stderr):
        self.parser = None
        self.program = program
        self._version = version
        self.err_stream = err_stream

        self._build_parser()

    def _build_parser(self):
        self.parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)

        # version flag
        self.parser.add_argument("--version", action="version", version=self.version)

        # datasheet search term
        self.parser.add_argument("search_term", help="search term")

    def parse(self, args):
        if len(args) == 1:
            self.print_help(exit=True)

        # parse arguments
        namespace = self.parser.parse_args(sys.argv[1:])

        # conduct action
        self.action(namespace)

    def action(self, namespace):
        request = DatasheetRequest(namespace.search_term)

    def print_help(self, exit=False):
        self.parser.print_help(self.err_stream)

        if exit:
            # exit with error code
            sys.exit(1)

    @property
    def version(self):
        return "{prog} {version}".format(prog=self.program, version=self._version)


def main():
    """Main program"""
    parser = Parser(PROGRAM, __version__)
    parser.parse(sys.argv)

if __name__ == "__main__":
    main()
