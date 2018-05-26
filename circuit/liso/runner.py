"""Tools for running LISO directly"""

import sys
import os
import logging
from tempfile import NamedTemporaryFile
import subprocess

from .output import LisoOutputParser

LOGGER = logging.getLogger("liso")

class Runner(object):
    """LISO runner"""

    def __init__(self, script_path):
        self.script_path = script_path

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

        return LisoOutputParser(output_path)

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
            raise Exception("error during LISO run: %s" % result.stderr)
        
        return result

    @property
    def liso_path(self):
        if self._liso_path is not None:
            return self._liso_path

        # use environment variable
        try:
            liso_dir = os.environ["LISO_DIR"]
        except KeyError:
            raise Exception("environment variable \"LISO_DIR\" must point to the "
                            "directory containing the LISO binary")

        return self.find_liso(liso_dir)

    @liso_path.setter
    def liso_path(self, path):
        self._liso_path = path

    @staticmethod
    def find_liso(directory):
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            # in order of preference
            filenames = ["fil_static", "fil"]
        elif sys.platform.startswith("win32"):
            filenames = ["fil.exe"]
        else:
            raise EnvironmentError("unrecognised operating system")

        for filename in filenames:
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                return path

        raise FileNotFoundError("no appropriate LISO binary found")