#!/usr/bin/env python

"""Datasheet utility"""

import sys
import logging
import argparse

from . import __version__, PROGRAM, DESCRIPTION, logging_on
from .request import DatasheetRequest

LOGGER = logging.getLogger(__name__)

class Parser(object):
    def __init__(self, program, version, info_stream=sys.stdout, err_stream=sys.stderr):
        self.parser = None
        self.program = program
        self._version = version
        self.info_stream = info_stream
        self.err_stream = err_stream

        self._build_parser()

    def _build_parser(self):
        self.parser = argparse.ArgumentParser(prog=PROGRAM, description=DESCRIPTION)

        # version flag
        self.parser.add_argument("--version", action="version", version=self.version)

        # verbose flag
        self.parser.add_argument("-v", "--verbose", action="store_true",
                                 help="enable verbose output")

        # datasheet search term
        self.parser.add_argument("term", help="search term")

        # force first result
        self.parser.add_argument("-f", "--first", action="store_true",
                                 help="display first part without prompt")

        # download and display
        self.parser.add_argument("-d", "--display", action="store_true",
                                 help="download and display datasheet")

        # no wildcards
        self.parser.add_argument("-e", "--exact", action="store_true",
                                 help="don't add wildcard characters around search term")

    def parse(self, args):
        if len(args) == 1:
            self.print_help(exit=True)

        # parse arguments
        namespace = self.parser.parse_args(sys.argv[1:])

        # conduct action
        self.action(namespace)

    def action(self, namespace):
        if namespace.verbose:
            # turn on logging
            logging_on()

        datasheets = DatasheetRequest(namespace.term, exact=namespace.exact)

        self._handle_parts(datasheets, first=namespace.first, display=namespace.display)

    def _handle_parts(self, datasheets, first=False, display=True):
        if datasheets.n_parts == 0:
            self.error("No parts found")
        elif datasheets.n_parts == 1 or first:
            # one datasheet
            datasheet = datasheets.latest_datasheet

            # show results directly
            self.info(datasheet)
            self._handle_part(datasheet, first=first, display=display)
        else:
            self.info("Found multiple parts:")
            for index, part in enumerate(datasheets.parts, 1):
                self.info("%d: %s" % (index, part))

            chosen_part_idx = 0
            while chosen_part_idx <= 0 or chosen_part_idx > datasheets.n_parts:
                try:
                    chosen_part_idx = int(input("Enter part number: "))

                    if chosen_part_idx <= 0 or chosen_part_idx > datasheets.n_parts:
                        raise ValueError
                except ValueError:
                    self.error("invalid, try again", exit=False)

            self._handle_part(datasheets.parts[chosen_part_idx - 1])

    def _handle_part(self, part, first=False, display=True):
        if part.n_datasheets == 0:
            self.error("No datasheets found for '%s'" % part.mpn)
        elif part.n_datasheets == 1 or first:
            # show results directly
            self.info(part)
            self._handle_datasheet(part.latest_datasheet, display=display)
        else:
            self.info("Found multiple datasheets:")
            for index, datasheet in enumerate(part.sorted_datasheets, 1):
                self.info("%d: %s" % (index, datasheet))

            chosen_datasheet_idx = 0
            while chosen_datasheet_idx <= 0 or chosen_datasheet_idx > part.n_datasheets:
                try:
                    chosen_datasheet_idx = int(input("Enter datasheet number: "))

                    if chosen_datasheet_idx <= 0 or chosen_datasheet_idx > part.n_datasheets:
                        raise ValueError
                except ValueError:
                    self.error("invalid, try again", exit=False)

            self._handle_datasheet(part.datasheets[chosen_datasheet_idx - 1])

    def _handle_datasheet(self, datasheet, display=True):
        self.info("Created: %s" % datasheet.created)
        self.info("Pages: %d" % datasheet.n_pages)
        self.info("URL: %s" % datasheet.url)

        if display:
            # download and display
            datasheet.display()

    def print_help(self, exit=False):
        self.parser.print_help(self.err_stream)

        if exit:
            # exit with error code
            sys.exit(1)

    def info(self, msg, exit=False):
        print(msg, file=self.info_stream)

        if exit:
            sys.exit(0)

    def error(self, msg, exit=True):
        print(msg, file=self.err_stream)

        if exit:
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
