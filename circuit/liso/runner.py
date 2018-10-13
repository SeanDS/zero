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
    """LISO runner

    Parameters
    ----------
    script_path : :class:`str`
        Path to LISO script to run.
    """
    def __init__(self, script_path):
        self.script_path = script_path

    def run(self, liso_path, output_path=None, plot=False, parse_output=True):
        """Run LISO script using a local LISO binary and handle the results

        Parameters
        ----------
        liso_path : :class:`str`
            Path to local LISO binary.
        output_path : :class:`str`, optional
            Path to save LISO output file to.
        plot : :class:`bool`, optional
            Plot the results using LISO.
        parse_output : :class:`bool`, optional
            Parse the output from LISO.

        Returns
        -------
        :class:`.LisoOutputParser`
            The parsed LISO output.
        """
        if output_path is None:
            # use temporary file
            temp_file = NamedTemporaryFile()
            output_path = temp_file.name

        # run LISO
        self._run_liso_process(liso_path, output_path, plot)

        if parse_output:
            parser = LisoOutputParser()
            parser.parse(output_path)
        else:
            parser = None

        return parser

    def _run_liso_process(self, liso_path, output_path, plot):
        input_path = os.path.abspath(self.script_path)

        if not os.path.exists(input_path):
            raise Exception("input file %s does not exist" % input_path)

        # LISO flags
        flags = [input_path, output_path]

        # plotting
        if not plot:
            flags.append("-n")

        LOGGER.debug("running LISO binary at %s", liso_path)

        # run LISO
        result = subprocess.run([liso_path, *flags], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise LisoError(result.stderr, script_path=self.script_path)

        return result


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
