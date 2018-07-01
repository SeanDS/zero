"""Configuration parser and defaults"""

import os.path
import abc
import logging
import re
import numpy as np
from configparser import ConfigParser
import pkg_resources
import appdirs

from .format import Quantity

LOGGER = logging.getLogger(__name__)

class SingletonAbstractMeta(abc.ABCMeta):
    """Abstract singleton class"""

    # dict of active instances
    _SINGLETON_REGISTRY = {}

    def __call__(cls, *args, **kwargs):
        """Return instance of `cls` if alrady exists, otherwise create"""

        if cls not in cls._SINGLETON_REGISTRY:
            # create new instance
            cls._SINGLETON_REGISTRY[cls] = super().__call__(*args, **kwargs)
        
        return cls._SINGLETON_REGISTRY[cls]

class BaseConfig(ConfigParser, metaclass=SingletonAbstractMeta):
    """Abstract configuration class"""

    CONFIG_FILENAME = None
    DEFAULT_CONFIG_FILENAME = None

    def __init__(self, *args, **kwargs):
        """Instantiate a new BaseConfig"""

        super().__init__(*args, **kwargs)

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
            pkg_resources.resource_filename(__name__, self.DEFAULT_CONFIG_FILENAME)
        )

    def load_user_config_file(self):
        """Load and parse a user config file"""

        config_file = self.get_user_config_filepath()

        # check the config file exists
        if os.path.isfile(config_file):
            self.load_config_file(config_file)

    @classmethod
    def get_user_config_filepath(cls):
        """Find the path to the config file

        This creates the config file if it does not exist, using the distributed
        template.

        :return: path to user config file
        :rtype: str
        """

        config_dir = appdirs.user_config_dir("circuit.py")
        config_file = os.path.join(config_dir, cls.CONFIG_FILENAME)

        return config_file

class CircuitConfig(BaseConfig):
    """Circuit config parser"""

    CONFIG_FILENAME = "circuit.conf"
    DEFAULT_CONFIG_FILENAME = CONFIG_FILENAME + ".dist"

class OpAmpLibrary(BaseConfig):
    CONFIG_FILENAME = "library.conf"
    DEFAULT_CONFIG_FILENAME = CONFIG_FILENAME + ".dist"

    # compiled regular expressions for parsing op-amp data
    COMMENT_REGEX = re.compile(r"\s*\#.*$")

    def __init__(self, *args, **kwargs):
        """Instantiate a new op-amp library"""

        # call parent constructor
        super().__init__(*args, **kwargs)

        # default options
        self.data = {}
        self.loaded = False

        # load and parse op-amp data from config file
        self.populate_library()

    def populate_library(self):
        """Load and parse op-amp data from config file"""

        count = 0

        # each section is a new op-amp
        for opamp in self.sections():
            self._parse_lib_data(opamp)
            count += 1

        LOGGER.debug("found %i op-amps", count)

        self.loaded = True

    @classmethod
    def format_name(cls, name):
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

        model = self.format_name(name)

        try:
            return self.data[model]
        except KeyError:
            raise ValueError("op-amp model %s not found in library" % name)

    def has_data(self, name):
        """Check if op-amp data exists in library

        :param name: op-amp name
        :type name: str
        :return: whether op-amp exists
        :rtype: bool
        """

        return self.format_name(name) in self.data.keys()

    def match(self, opamp):
        """Get model name of library op-amp given a specified op-amp

        :param opamp: op-amp object to match
        :type opamp: :class:`~OpAmp`
        :return: model name, or None
        :rtype: str or NoneType
        """

        for model in self.data:
            if opamp.params == self.data[model]:
                return model

        return None

    def _parse_lib_data(self, section):
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
            class_data["a0"] = Quantity(self._strip_comments(opamp_data["a0"]))
        if "gbw" in opamp_data:
            class_data["gbw"] = Quantity(self._strip_comments(opamp_data["gbw"]), "Hz")
        if "delay" in opamp_data:
            class_data["delay"] = Quantity(self._strip_comments(opamp_data["delay"]), "s")
        if "vn" in opamp_data:
            class_data["v_noise"] = Quantity(self._strip_comments(opamp_data["vn"]), "V/sqrt(Hz)")
        if "in" in opamp_data:
            class_data["i_noise"] = Quantity(self._strip_comments(opamp_data["in"]), "A/sqrt(Hz)")
        if "vc" in opamp_data:
            class_data["v_corner"] = Quantity(self._strip_comments(opamp_data["vc"]), "Hz")
        if "ic" in opamp_data:
            class_data["i_corner"] = Quantity(self._strip_comments(opamp_data["ic"]), "Hz")
        if "vmax" in opamp_data:
            class_data["v_max"] = Quantity(self._strip_comments(opamp_data["vmax"]), "V")
        if "imax" in opamp_data:
            class_data["i_max"] = Quantity(self._strip_comments(opamp_data["imax"]), "A")
        if "sr" in opamp_data:
            class_data["slew_rate"] = Quantity(self._strip_comments(opamp_data["sr"]), "V/s")

        # add data to library
        self.add_data(section, class_data)

        # check if there are aliases
        if "aliases" in opamp_data:
            # get individual aliases
            aliases = [alias.strip() for alias
                       in opamp_data["aliases"].split(",")]

            # create new op-amps for each alias using identical data
            for alias in aliases:
                self.add_data(alias, class_data)

    def add_data(self, name, data):
        """Add op-amp data to library

        :param name: op-amp name
        :type name: str
        :param data: op-amp data
        :type data: Dict[str, Any]
        :raises ValueError: if op-amp is already in library
        """

        name = self.format_name(name)

        if name in self.opamp_names:
            raise ValueError("Duplicate op-amp type: %s" % name)

        # set data
        self.data[name] = data

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

        This also strips out comments.

        :param entry: list of frequencies to split
        :type entry: str
        :return: array of complex frequencies
        :rtype: :class:`~np.array`
        """

        # strip out comments
        entry = self._strip_comments(entry)

        # split into groups delimited by commas
        freq_tokens = [freq.strip() for freq in entry.split(",")]

        # generate complex frequencies from the list and combine them into one list
        frequencies = []
        
        for token in freq_tokens:
            frequencies.extend(self._parse_freq_str(token))

        return np.array(frequencies)

    def _parse_freq_str(self, token):
        """Parse token as complex frequency/frequencies

        The frequency may include an optional q-factor, which results in this
        method returning a pair of equal and opposite complex frequencies. The
        one or two returned frequencies are always contained in a list.

        :param token: string containing frequency and optional q-factor
        :type token: str
        :return: list of frequencies
        :rtype: List[Numpy scalar or float]
        """

        frequencies = []

        # split frequency and optional q-factor into list entries
        parts = token.split()

        # frequency is always first in the list
        frequency = Quantity(parts[0], "Hz")

        # q-factor is second, if present
        if len(parts) == 1:
            frequencies.append(frequency)
        elif len(parts) == 2:
            # calculate complex frequency using q-factor
            qfactor = Quantity(parts[1])
            # cast to complex to avoid issues with arccos
            qfactor = complex(qfactor)
            theta = np.arccos(1 / (2 * qfactor))

            # add negative/positive pair of poles/zeros
            frequencies.append(frequency * np.exp(-1j * theta))
            frequencies.append(frequency * np.exp(1j * theta))
        else:
            raise Exception("invalid frequency list")

        return frequencies
