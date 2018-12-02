"""Configuration parser and defaults"""

from .base import BaseConfig


class ZeroConfig(BaseConfig):
    """Zero config parser"""
    # user config filename
    USER_CONFIG_FILENAME = "zero.yaml"
    # default config copied to user directory if requested
    DEFAULT_USER_CONFIG_FILENAME = USER_CONFIG_FILENAME + ".dist"
    # config into which others are merged
    BASE_CONFIG_FILENAME = USER_CONFIG_FILENAME + ".dist.default"
