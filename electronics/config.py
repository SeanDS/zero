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
    CONFIG_FILENAME = None
    DEFAULT_CONFIG_FILENAME = None

    def __init__(self, *args, **kwargs):
        super(BaseConfig, self).__init__(*args, **kwargs)

        self.load_default_config_file()
        self.load_user_config_file()

    def load_config_file(self, path):
        with open(path) as obj:
            LOGGER.debug("reading config from %s", path)
            self.read_file(obj)

    def load_default_config_file(self):
        self.load_config_file(
            pkg_resources.resource_filename(__name__,
                                            self.DEFAULT_CONFIG_FILENAME)
        )

    def load_user_config_file(self):
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
        """

        config_dir = appdirs.user_config_dir("electronics")
        config_file = os.path.join(config_dir, cls.CONFIG_FILENAME)

        return config_file

    @classmethod
    def create_user_config_file(cls, config_file):
        """Create config file in user directory"""

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

    COMMENT_REGEX = re.compile("\s*\#.*$")
    BRACKET_REGEX = re.compile("\[([\w\d\s.+\-]*)\]")

    def __init__(self, *args, **kwargs):
        super(OpAmpLibrary, self).__init__(*args, **kwargs)

        self.data = {}
        self.loaded = False

        self.populate_library()

    def populate_library(self):
        for opamp in self.sections():
            self._parse_config(opamp)

        self.loaded = True

    def get_data(self, name):
        return self.data[name.upper()]

    def has_data(self, name):
        return name.upper() in self.data.keys()

    def _parse_config(self, section):
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

        class_data = {"zeros": zeros, "poles": poles}

        if "a0" in opamp_data:
            class_data["a0"] = self._clean_parse_entry(opamp_data["a0"])
        if "gbw" in opamp_data:
            class_data["gbw"] = self._clean_parse_entry(opamp_data["gbw"])
        if "delay" in opamp_data:
            class_data["delay"] = self._clean_parse_entry(opamp_data["delay"])
        if "vn" in opamp_data:
            class_data["vn"] = self._clean_parse_entry(opamp_data["vn"])
        if "in" in opamp_data:
            class_data["in"] = self._clean_parse_entry(opamp_data["in"])
        if "vc" in opamp_data:
            class_data["vc"] = self._clean_parse_entry(opamp_data["vc"])
        if "ic" in opamp_data:
            class_data["ic"] = self._clean_parse_entry(opamp_data["ic"])
        if "vmax" in opamp_data:
            class_data["vmax"] = self._clean_parse_entry(opamp_data["vmax"])
        if "imax" in opamp_data:
            class_data["imax"] = self._clean_parse_entry(opamp_data["imax"])
        if "sr" in opamp_data:
            class_data["sr"] = self._clean_parse_entry(opamp_data["sr"])

        # add to library
        self.add_data(section, class_data)

        # check if there are aliases
        if "aliases" in opamp_data:
            for alias_name in opamp_data["aliases"].split():
                self.add_data(alias_name, class_data)

    def add_data(self, name, data):
        name = name.upper()

        if name in self.opamp_names:
            raise Exception("Duplicate op-amp type: %s" % name)

        LOGGER.debug("adding op-amp data for %s", name)

        self.data[name] = data

    def _clean_parse_entry(self, entry):
        return SIFormatter.parse(self._strip_comments(entry))

    @property
    def opamp_names(self):
        return self.data.keys()

    def _strip_comments(self, line):
        return re.sub(self.COMMENT_REGEX, "", line)

    def _split_frequencies(self, line):
        """Splits lists of frequencies into groups

        Will split e.g. "[7.5M 1.78] [1.6M]" into ["7.5M 1.78", "1.6M"]

        This also strips out comments.
        """
        return re.findall(self.BRACKET_REGEX, line)

    def _parse_freq_set(self, freq_set):
        # split into groups defined by square brackets
        freq_tokens = self._split_frequencies(freq_set)

        values = []

        for token in freq_tokens:
            value = self._parse_freq_str(token)

            if value is not None:
                values.extend(value)

        return np.array(values)

    def _parse_freq_str(self, token):
        # values
        values = []

        # split frequency and q-factor (if present)
        parts = token.split()
        frequency = SIFormatter.parse(parts[0])

        if len(parts) > 1:
            # complex
            qfactor = SIFormatter.parse(parts[1])
            theta = np.arccos(1 / (2 * qfactor))

            # negative/positive poles/zeros
            values.append(frequency * np.exp(-1j * theta))
            values.append(frequency * np.exp(1j * theta))
        else:
            values.append(frequency)

        return values
