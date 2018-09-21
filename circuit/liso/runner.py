"""Tools for running LISO directly"""

import sys
import os
import logging
from tempfile import NamedTemporaryFile
import subprocess
import shutil

from .base import LisoParserError
from .output import LisoOutputParser

LOGGER = logging.getLogger(__name__)


class LisoRunner:
    """LISO runner"""

    # LISO binary names, in order of preference
    LISO_BINARY_NAMES = {"nix": ["fil_static", "fil"], # Linux / OSX
                         "win": ["fil.exe"]}           # Windows

    def __init__(self, script_path=None):
        self.script_path = script_path

        # defaults
        self._liso_path = None

    def run(self, liso_plot=False, liso_parse=True, liso_path=None, output_path=None):
        """Run LISO script using a local LISO binary and handle the results

        Parameters
        ----------
        liso_plot : :class:`bool`, optional
            Plot the results using LISO.
        liso_parse : :class:`bool`, optional
            Parse the output from LISO.
        liso_path : :class:`str`, optional
            Path to local LISO binary. If not specified, this program will attempt to find it
            automatically.
        output_path : :class:`str`, optional
            Path to save LISO output file to. If not specified, LISO's default is used.

        Returns
        -------
        :class:`.LisoOutputParser`
            The parsed LISO output.
        """
        self.liso_path = liso_path

        if not output_path:
            temp_file = NamedTemporaryFile()
            output_path = temp_file.name

        return self._liso_result(self.script_path, output_path, liso_parse, liso_plot)

    def _liso_result(self, script_path, output_path, parse, plot):
        """Get LISO results

        Parameters
        ----------
        script_path : :class:`str`
            Path to the LISO ".fil" file.
        output_path : :class:`str`
            Path to LISO ".out" file to be created.
        parse : :class:`bool`
            Parse the output from LISO.
        plot : :class:`bool`
            Ask LISO to plot the result instead of this program.

        Returns
        -------
        :class:`.OutputParser` or None
            Parsed LISO output, if requested, otherwise None.
        """
        self._run_liso_process(script_path, output_path, plot)

        if parse:
            parser = LisoOutputParser()
            parser.parse(path=output_path)
        else:
            parser = None

        return parser

    def _run_liso_process(self, script_path, output_path, plot):
        input_path = os.path.abspath(script_path)

        if not os.path.exists(input_path):
            raise Exception("input file %s does not exist" % input_path)

        # LISO flags
        flags = [input_path, output_path]

        # plotting
        if not plot:
            flags.append("-n")

        liso_path = self.liso_path
        LOGGER.debug("running LISO binary at %s", liso_path)

        # run LISO
        result = subprocess.run([liso_path, *flags], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise LisoError(result.stderr, script_path=script_path)

        return result

    @property
    def liso_path(self):
        if self._liso_path is not None:
            return self._liso_path

        liso_path = None

        # try environment variable
        try:
            liso_path = self.find_liso(os.environ["LISO_DIR"])
        except (KeyError, FileNotFoundError):
            # no environment variable set or LISO not found in specified directory
            # try searching path
            for command in self.LISO_BINARY_NAMES[self.platform_key]:
                path = shutil.which(command)
                if path is not None:
                    liso_path = path
                    break

        if liso_path is None:
            raise FileNotFoundError("environment variable \"LISO_DIR\" must point to "
                                    "the directory containing the LISO binary, or the "
                                    "LISO binary must be available on the system PATH "
                                    "(and executable)")

        return liso_path

    @liso_path.setter
    def liso_path(self, path):
        self._liso_path = path

    @property
    def platform_key(self):
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            return "nix"
        elif sys.platform.startswith("win32"):
            return "win"

        raise EnvironmentError("unrecognised operating system")

    def find_liso(self, directory):
        for filename in self.LISO_BINARY_NAMES[self.platform_key]:
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                return path

        raise FileNotFoundError("no appropriate LISO binary found")


class LisoError(Exception):
    def __init__(self, message, script_path=None, *args, **kwargs):
        """LISO error

        Parameters
        ----------
        message : :class:`str` or :class:`bytes`
            The error message, or `sys.stderr` bytes buffer.
        script_path : :class:`str`, optional
            The path to the script that caused the error (used to check for common mistakes).
        """
        if isinstance(message, bytes):
            # decode stderr bytes
            message = self._parse_liso_error(message.decode("utf-8"))

        if script_path is not None:
            if os.path.isfile(script_path):
                parser = LisoOutputParser()

                # attempt to parse as input
                try:
                    parser.parse(path=script_path)

                    is_output = True
                except (IOError, LisoParserError):
                    is_output = False

                if is_output:
                    # add message
                    message = "{message} (this appears to be a LISO output file)".format(message=message)

        super().__init__(message, *args, **kwargs)

    def _parse_liso_error(self, error_msg):
        # split into lines
        lines = error_msg.splitlines()

        for line in lines:
            line = line.strip()
            if line.startswith("*** Error:"):
                # return error
                return line.lstrip("*** Error:")

        return "[error message not detected] LISO output:\n%s" % "\n".join(lines)
