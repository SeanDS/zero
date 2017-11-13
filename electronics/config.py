"""Configuration parser and defaults"""

import os.path
import abc
import logging
from configparser import RawConfigParser
import pkg_resources
import appdirs

# logger
logger = logging.getLogger("config")

class BaseConfig(RawConfigParser, metaclass=abc.ABCMeta):
    CONFIG_FILENAME = None
    DEFAULT_CONFIG_FILENAME = None

    def __init__(self, *args, **kwargs):
        super(BaseConfig, self).__init__(*args, **kwargs)

        self.load_default_config_file()
        self.load_user_config_file()

    def load_config_file(self, path):
        with open(path) as obj:
            logger.debug("Reading config from %s", path)
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

        logger.debug("Creating empty config file at %s", directory)

        # touch file
        open(config_file, 'w').close()

class ElectronicsConfig(BaseConfig):
    """Electronics config parser"""

    CONFIG_FILENAME = "electronics.conf"
    DEFAULT_CONFIG_FILENAME = CONFIG_FILENAME + ".dist"
