"""Configuration parser and defaults"""

import os.path
import abc
import logging
import re
from configparser import ConfigParser
import numpy as np
import pkg_resources
import click

from . import PROGRAM
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

        config_dir = click.get_app_dir(PROGRAM)
        config_file = os.path.join(config_dir, cls.CONFIG_FILENAME)

        return config_file


class ZeroConfig(BaseConfig):
    """Zero config parser"""

    CONFIG_FILENAME = "zero.conf"
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
        if "vnoise" in opamp_data:
            class_data["vnoise"] = Quantity(self._strip_comments(opamp_data["vnoise"]), "V/sqrt(Hz)")
        if "vcorner" in opamp_data:
            class_data["vcorner"] = Quantity(self._strip_comments(opamp_data["vcorner"]), "Hz")
        if "inoise" in opamp_data:
            class_data["inoise"] = Quantity(self._strip_comments(opamp_data["inoise"]), "A/sqrt(Hz)")
        if "icorner" in opamp_data:
            class_data["icorner"] = Quantity(self._strip_comments(opamp_data["icorner"]), "Hz")
        if "vmax" in opamp_data:
            class_data["vmax"] = Quantity(self._strip_comments(opamp_data["vmax"]), "V")
        if "imax" in opamp_data:
            class_data["imax"] = Quantity(self._strip_comments(opamp_data["imax"]), "A")
        if "sr" in opamp_data:
            class_data["sr"] = Quantity(self._strip_comments(opamp_data["sr"]), "V/s")

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

    @property
    def opamps(self):
        opdata = self.data.items()

        return [LibraryOpAmp(model=self.format_name(model), **data) for model, data in opdata]


class LibraryOpAmp:
    """Represents a library op-amp.

    Some of the default parameter values are based on the OP27.

    Parameters
    ----------
    model : :class:`str`
        Model name.
    a0 : :class:`float`, optional
        Open loop gain.
    gbw : :class:`float`, optional
        Gain-bandwidth product.
    delay : :class:`float`, optional
        Delay.
    zeros : sequence, optional
        Zeros.
    poles : sequence, optional
        Poles.
    vnoise : :class:`float`, optional
        Flat voltage noise.
    vcorner : :class:`float`, optional
        Voltage noise corner frequency.
    inoise : :class:`float`, optional
        Float current noise.
    icorner : :class:`float`, optional
        Current noise corner frequency.
    vmax : :class:`float`, optional
        Maximum input voltage.
    imax : :class:`float`, optional
        Maximum output current.
    sr : :class:`float`, optional
        Slew rate.
    """
    def __init__(self, model="OP00", a0=1.5e6, gbw=8e6, delay=0, zeros=np.array([]),
                 poles=np.array([]), vnoise=3.2e-9, vcorner=2.7, inoise=0.4e-12, icorner=140,
                 vmax=12, imax=0.06, sr=1e6, **kwargs):
        super().__init__(**kwargs)

        # default properties
        self._model = "None"
        self.params = {}

        # set parameters
        self.model = model
        self.a0 = a0
        self.gbw = gbw
        self.delay = delay
        self.zeros = zeros
        self.poles = poles
        self.vnoise = vnoise
        self.vcorner = vcorner
        self.inoise = inoise
        self.icorner = icorner
        self.vmax = vmax
        self.imax = imax
        self.sr = sr

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = str(model).upper()

    @property
    def a0(self):
        """Gain"""
        return self.params["a0"]

    @a0.setter
    def a0(self, a0):
        self.params["a0"] = Quantity(a0)

    @property
    def gbw(self):
        """Gain-bandwidth product"""
        return self.params["gbw"]

    @gbw.setter
    def gbw(self, gbw):
        self.params["gbw"] = Quantity(gbw, "Hz")

    @property
    def delay(self):
        """Delay"""
        return self.params["delay"]

    @delay.setter
    def delay(self, delay):
        self.params["delay"] = Quantity(delay, "s")

    @property
    def zeros(self):
        """Additional zeros"""
        return self.params["zeros"]

    @zeros.setter
    def zeros(self, zeros):
        self.params["zeros"] = np.array(zeros)

    @property
    def poles(self):
        """Additional poles"""
        return self.params["poles"]

    @poles.setter
    def poles(self, poles):
        self.params["poles"] = np.array(poles)

    @property
    def vnoise(self):
        """Voltage noise"""
        return self.params["vnoise"]

    @vnoise.setter
    def vnoise(self, vnoise):
        self.params["vnoise"] = Quantity(vnoise, "V/sqrt(Hz)")

    @property
    def vcorner(self):
        """Voltage noise corner frequency"""
        return self.params["vcorner"]

    @vcorner.setter
    def vcorner(self, vcorner):
        self.params["vcorner"] = Quantity(vcorner, "Hz")

    @property
    def inoise(self):
        """Current noise"""
        return self.params["inoise"]

    @inoise.setter
    def inoise(self, inoise):
        self.params["inoise"] = Quantity(inoise, "A/sqrt(Hz)")

    @property
    def icorner(self):
        """Current noise corner frequency"""
        return self.params["icorner"]

    @icorner.setter
    def icorner(self, icorner):
        self.params["icorner"] = Quantity(icorner, "Hz")

    @property
    def vmax(self):
        """Maximum output voltage"""
        return self.params["vmax"]

    @vmax.setter
    def vmax(self, vmax):
        self.params["vmax"] = Quantity(vmax, "V")

    @property
    def imax(self):
        """Maximum output current"""
        return self.params["imax"]

    @imax.setter
    def imax(self, imax):
        self.params["imax"] = Quantity(imax, "A")

    @property
    def sr(self):
        """Slew rate"""
        return self.params["sr"]

    @sr.setter
    def sr(self, sr):
        self.params["sr"] = Quantity(sr, "V/s")

    def gain(self, frequency):
        """Get op-amp voltage gain at the specified frequency.

        Parameters
        ----------
        frequency : :class:`float`
            Frequency to compute gain at.

        Returns
        -------
        :class:`float`
            Op-amp gain at specified frequency.
        """

        return (self.a0
                / (1 + self.a0 * 1j * frequency / self.gbw)
                * np.exp(-2j * np.pi * self.delay * frequency)
                * np.prod(1 + 1j * frequency / self.zeros)
                / np.prod(1 + 1j * frequency / self.poles))

    def inverse_gain(self, *args, **kwargs):
        """Op-amp inverse gain.

        Note that the inverse gain may be modified by the analysis, e.g. in the
        case of a voltage follower (see :meth:`zero.analysis.ac.BaseAcAnalysis.component_equation`).
        """
        return 1 / self.gain(*args, **kwargs)

    def __str__(self):
        return "{cmp.model}(a0={cmp.a0}, gbw={cmp.gbw}, delay={cmp.delay})".format(cmp=self)
