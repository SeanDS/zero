"""Base configuration parser"""

import os.path
import shutil
import logging
import pkg_resources
import click
from yaml import safe_load
from click import launch

from .. import PROGRAM
from ..misc import Singleton

LOGGER = logging.getLogger(__name__)


class BaseConfig(dict, metaclass=Singleton):
    """Base YAML config parser"""
    # user config filename
    USER_CONFIG_FILENAME = None
    # default config copied to user directory if requested
    DEFAULT_USER_CONFIG_FILENAME = None
    # config into which others are merged
    BASE_CONFIG_FILENAME = None

    def __init__(self):
        # flag whether config is invalid
        self.user_config_invalid = False

        # load default config then override with user config
        self._load_base_config()
        self._load_user_config()

    @property
    def base_config_path(self):
        return pkg_resources.resource_filename(__name__, self.BASE_CONFIG_FILENAME)

    @property
    def user_config_path(self):
        config_dir = click.get_app_dir(PROGRAM)
        return os.path.join(config_dir, self.USER_CONFIG_FILENAME)

    def create_user_config(self):
        if os.path.exists(self.user_config_path):
            raise ConfigAlreadyExistsException(self.user_config_path)

        LOGGER.info("Creating default user config file at %s", self.user_config_path)

        # copy default
        default_path = pkg_resources.resource_filename(__name__, self.DEFAULT_USER_CONFIG_FILENAME)

        # user configuration directory
        config_dir = os.path.dirname(self.user_config_path)

        if not os.path.isdir(config_dir):
            # make directory tree
            os.makedirs(config_dir)

        # copy default config into user config directory
        shutil.copyfile(default_path, self.user_config_path)

    def open_user_config(self):
        if not os.path.isfile(self.user_config_path):
            raise ConfigDoesntExistException(self.user_config_path)

        launch(self.user_config_path)

    def remove_user_config(self):
        try:
            os.remove(self.user_config_path)
        except FileNotFoundError:
            raise ConfigDoesntExistException(self.user_config_path)

    def _load_base_config(self):
        self._merge_yaml_file(self.base_config_path)

    def _load_user_config(self):
        """Load user config file"""
        # check the config file exists
        if not os.path.isfile(self.user_config_path):
            LOGGER.info("No user config file found at %s", self.user_config_path)
            return

        try:
            self._merge_yaml_file(self.user_config_path)
        except Exception as e:
            # an error occurred loading user file
            if not self.user_config_invalid:
                LOGGER.error("user config file at %s is invalid", self.user_config_path)
                self.user_config_invalid = True

    def _merge_yaml_file(self, path):
        with open(path, "r") as configfile:
            config = safe_load(configfile)

        if config is None:
            # config may be empty
            LOGGER.debug("config file at %s is empty", path)
            return

        self._merge_config(config)

    def _merge_config(self, config):
        """Merge specified configuration with this one."""
        self._merge_recursive(self, config)

    @classmethod
    def _merge_recursive(cls, config_a, config_b, path=None):
        """Merge second configuration into first configuration.
        https://stackoverflow.com/a/7205107
        """
        if path is None:
            path = []

        for key in config_b:
            if key in config_a:
                if isinstance(config_a[key], dict) and isinstance(config_b[key], dict):
                    # merge values together
                    cls._merge_recursive(config_a[key], config_b[key], path + [str(key)])
                elif config_a[key] == config_b[key]:
                    # same leaf value
                    pass
                elif config_b[key] is None:
                    # don't copy anything to preserve a's structure
                    pass
                else:
                    dpath = '.'.join(path + [str(key)])
                    raise Exception(f"configuration conflict at {dpath}")
            else:
                config_a[key] = config_b[key]

        return config_a


class ConfigDoesntExistException(Exception):
    def __init__(self, config_path, *args, **kwargs):
        super().__init__(f"config file {config_path} doesn't exist", *args, **kwargs)


class ConfigAlreadyExistsException(Exception):
    def __init__(self, config_path, *args, **kwargs):
        super().__init__(f"config file already exists at {config_path}", *args, **kwargs)
