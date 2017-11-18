"""Configuration parser and defaults"""

import os.path
import abc
import logging
import re
import numpy as np
from configparser import RawConfigParser
import pkg_resources
import appdirs

from .format import SIFormatter

LOGGER = logging.getLogger("config")

class BaseConfig(RawConfigParser, metaclass=abc.ABCMeta):
    """Abstract configuration class"""

    CONFIG_FILENAME = None
    DEFAULT_CONFIG_FILENAME = None

    def __init__(self, *args, **kwargs):
        """Instantiate a new BaseConfig"""

        super(BaseConfig, self).__init__(*args, **kwargs)

        # load default config then overwrite with user config if present
        self.load_default_config_file()
        self.load_user_config_file()

    def load_config_file(self, path):
        """Load and parse a config file

        :param path: config file path
        :type path: str
        """

        with open(path) as obj:
            LOGGER.debug("reading config from %s", path)
            self.read_file(obj)

    def load_default_config_file(self):
        """Load and parse the default config file"""

        self.load_config_file(
            pkg_resources.resource_filename(__name__,
                                            self.DEFAULT_CONFIG_FILENAME)
        )

    def load_user_config_file(self):
        """Load and parse a user config file"""

        config_file = self.get_user_config_filepath()

        # check the config file exists
        if not os.path.isfile(config_file):
            self.create_user_config_file(config_file)

        self.load_config_file(config_file)

    @classmethod
    def get_user_config_filepath(cls):
        """Find the path to the config file

        This creates the config file if it does not exist, using the distributed
        template.

        :return: path to user config file
        :rtype: str
        """

        config_dir = appdirs.user_config_dir("electronics")
        config_file = os.path.join(config_dir, cls.CONFIG_FILENAME)

        return config_file

    @classmethod
    def create_user_config_file(cls, config_file):
        """Create empty config file in user directory

        :param config_file: path to config file
        :type config_file: str
        """

        directory = os.path.dirname(config_file)

        # create user config directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        LOGGER.debug("creating empty config file at %s", directory)

        # touch file
        open(config_file, 'w').close()

class ElectronicsConfig(BaseConfig):
    """Electronics config parser"""

    CONFIG_FILENAME = "electronics.conf"
    DEFAULT_CONFIG_FILENAME = CONFIG_FILENAME + ".dist"

class OpAmpLibrary(BaseConfig):
    CONFIG_FILENAME = "library.conf"
    DEFAULT_CONFIG_FILENAME = CONFIG_FILENAME + ".dist"

    # compiled regular expressions for parsing op-amp data
    COMMENT_REGEX = re.compile("\s*\#.*$")
    BRACKET_REGEX = re.compile("\[([\w\d\s.+\-]*)\]")

    def __init__(self, *args, **kwargs):
        """Instantiate a new op-amp library"""

        # call parent constructor
        super(OpAmpLibrary, self).__init__(*args, **kwargs)

        # default options
        self.data = {}
        self.loaded = False

        # load and parse op-amp data from config file
        self.populate_library()

    def populate_library(self):
        """Load and parse op-amp data from config file"""

        # each section is a new op-amp
        for opamp in self.sections():
            self._parse_opamp_data(opamp)

        self.loaded = True

    def _format_name(self, name):
        """Format op-amp name for use as a key in the data dict

        :param name: name to format
        :type name: str
        :return: formatted name
        :rtype: str
        """

        return str(name).upper()

    def get_data(self, name):
        """Get op-amp data

        :param name: op-amp name
        :type name: str
        :return: op-amp data
        :rtype: dict
        """

        return self.data[self._format_name(name)]

    def has_data(self, name):
        """Check if op-amp data exists in library

        :param name: op-amp name
        :type name: str
        :return: whether op-amp exists
        :rtype: bool
        """

        return self._format_name(name) in self.data.keys()

    def _parse_opamp_data(self, section):
        """Parse op-amp data from config file

        :param section: section of config file correponding to op-amp
        :type section: str
        """

        opamp_data = self[section]

        # handle poles and zeros
        if "poles" in opamp_data:
            poles = self._parse_freq_set(opamp_data["poles"])
        else:
            poles = np.array([])

        if "zeros" in opamp_data:
            zeros = self._parse_freq_set(opamp_data["zeros"])
        else:
            zeros = np.array([])

        # build op-amp data dict with poles and zeros as entries
        class_data = {"zeros": zeros, "poles": poles}

        # add other op-amp data
        if "a0" in opamp_data:
            class_data["a0"] = self._clean_parse_param(opamp_data["a0"])
        if "gbw" in opamp_data:
            class_data["gbw"] = self._clean_parse_param(opamp_data["gbw"])
        if "delay" in opamp_data:
            class_data["delay"] = self._clean_parse_param(opamp_data["delay"])
        if "vn" in opamp_data:
            class_data["vn"] = self._clean_parse_param(opamp_data["vn"])
        if "in" in opamp_data:
            class_data["in"] = self._clean_parse_param(opamp_data["in"])
        if "vc" in opamp_data:
            class_data["vc"] = self._clean_parse_param(opamp_data["vc"])
        if "ic" in opamp_data:
            class_data["ic"] = self._clean_parse_param(opamp_data["ic"])
        if "vmax" in opamp_data:
            class_data["vmax"] = self._clean_parse_param(opamp_data["vmax"])
        if "imax" in opamp_data:
            class_data["imax"] = self._clean_parse_param(opamp_data["imax"])
        if "sr" in opamp_data:
            class_data["sr"] = self._clean_parse_param(opamp_data["sr"])

        # add data to library
        self.add_data(section, class_data)

        # check if there are aliases
        if "aliases" in opamp_data:
            # create new op-amps for each alias using identical data
            for alias_name in opamp_data["aliases"].split():
                self.add_data(alias_name, class_data)

    def add_data(self, name, data):
        """Add op-amp data to library

        :param name: op-amp name
        :type name: str
        :param data: op-amp data
        :type data: Dict[str, Dict[str, Any]]
        :raises ValueError: if op-amp is already in library
        """

        name = self._format_name(name)

        if name in self.opamp_names:
            raise ValueError("Duplicate op-amp type: %s" % name)

        LOGGER.debug("adding op-amp data for %s", name)

        # set data
        self.data[name] = data

    def _clean_parse_param(self, param):
        """Clean and parse as a float an op-amp config parameter

        :param param: parameter to clean and parse
        :type param: str
        :return: parsed parameter
        :rtype: float
        """

        return SIFormatter.parse(self._strip_comments(param))

    @property
    def opamp_names(self):
        """Get names of op-amps in library (including alises)

        :return: op-amp names
        :rtype: KeysView[str]
        """

        return self.data.keys()

    def _strip_comments(self, line):
        """Remove comments from specified config file line

        :param line: line to clean
        :type line: str
        :return: line with comments removed
        :rtype: str
        """

        return re.sub(self.COMMENT_REGEX, "", line)

    def _parse_freq_set(self, entry):
        """Parse string list of frequencies and q-factors as a numpy array

        :param entry: list of frequencies to split
        :type entry: str
        :return: array of complex frequencies
        :rtype: :class:`~np.array`
        """

        # split into groups defined by square brackets
        freq_tokens = self._split_frequencies(entry)

        values = []

        for token in freq_tokens:
            value = self._parse_freq_str(token)

            if value is not None:
                values.extend(value)

        return np.array(values)

    def _parse_freq_str(self, token):
        """Parse token as complex frequency/frequencies

        The frequency may include an optional q-factor, which results in this
        method returning a pair of equal and opposite complex frequencies. The
        one or two returned frequencies are always contained in a list.

        :param token: string containing frequency and optional q-factor
        :type token: str
        :return: list of frequencies
        :rtype: List[:class:`np.complex128` or float]
        """

        frequencies = []

        # split frequency and optional q-factor into list entries
        parts = token.split()

        # frequency is always first in the list
        frequency = SIFormatter.parse(parts[0])

        # q-factor is second, if present
        if len(parts) > 1:
            # calculate complex frequency using q-factor
            qfactor = SIFormatter.parse(parts[1])
            theta = np.arccos(1 / (2 * qfactor))

            # add negative/positive pair of poles/zeros
            frequencies.append(frequency * np.exp(-1j * theta))
            frequencies.append(frequency * np.exp(1j * theta))
        else:
            frequencies.append(frequency)

        return frequencies

    def _split_frequencies(self, entry):
        """Split list of frequencies into sequence of individual frequencies

        Will split e.g. "[7.5M 1.78] [1.6M]" into ["7.5M 1.78", "1.6M"].
        This also strips out comments.

        :param entry: list of frequencies to split
        :type entry: str
        :return: individual frequency strings
        :rtype: Sequence[str]
        """

        # split individual groups of frequencies and optional q-factors by
        # searching for groups of square brackets
        return re.findall(self.BRACKET_REGEX, entry)
