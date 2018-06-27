"""Tools for running LISO directly"""

import sys
import os
import logging
from tempfile import NamedTemporaryFile
import subprocess
import shutil

from .base import LisoParserError
from .output import LisoOutputParser

LOGGER = logging.getLogger("liso")

class LisoRunner(object):
    """LISO runner"""

    # LISO binary names, in order of preference
    LISO_BINARY_NAMES = {"nix": ["fil_static", "fil"], # Linux / OSX
                         "win": ["fil.exe"]}           # Windows

    def __init__(self, script_path=None):
        self.script_path = script_path

        # defaults
        self._liso_path = None

    def run(self, plot=False, liso_path=None, output_path=None):
        self.liso_path = liso_path

        if not output_path:
            temp_file = NamedTemporaryFile()
            output_path = temp_file.name

        return self._liso_result(self.script_path, output_path, plot)

    def _liso_result(self, script_path, output_path, plot):
        """Get LISO results

        :param script_path: path to LISO ".fil" file
        :type script_path: str
        :param output_path: path to LISO ".out" file to be created
        :type output_path: str
        :param plot: whether to show result with gnuplot
        :type plot: bool
        :return: LISO output
        :rtype: :class:`~OutputParser`
        """

        self._run_liso_process(script_path, output_path, plot)

        parser = LisoOutputParser()
        parser.parse(path=output_path)

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

    @property
    def platform_key(self):
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            return "nix"
        elif sys.platform.startswith("win32"):
            return "win"
        
        raise EnvironmentError("unrecognised operating system")

    @liso_path.setter
    def liso_path(self, path):
        self._liso_path = path

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
