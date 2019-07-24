"""Miscellaneous functions"""

import sys
import os
import abc
import tempfile
import numpy as np
import requests
import progressbar


class Singleton(abc.ABCMeta):
    """Metaclass implementing the singleton pattern

    This ensures that there is only ever one instance of a class that
    inherits this one.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by class
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


class NamedInstance(abc.ABCMeta):
    """Metaclass to implement a single named instance pattern

    This ensures that there is only ever one instance of a class with a specific
    name, as provided by the "name" constructor argument.

    This is a subclass of ABCMeta so that it can be used as a metaclass of a
    subclass of an ABCMeta class.
    """

    # list of children by name
    _names = {}

    def __call__(cls, name, *args, **kwargs):
        name = name.lower()

        if name not in cls._names:
            cls._names[name] = super().__call__(name, *args, **kwargs)

        return cls._names[name]


class ChangeFlagDict(dict):
    """Dict with flag which gets set when a change is made after initialisation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # change flag
        self.changed = False

    def __setitem__(self, *args, **kwargs):
        super().__setitem__(*args, **kwargs)
        self.changed = True

    def __delitem__(self, *args, **kwargs):
        super().__delitem__(*args, **kwargs)
        self.changed = True


class Downloadable:
    """Mixin for downloadable URL classes, providing a progress bar."""
    def __init__(self, info_stream=sys.stdout, progress=True, timeout=None, **kwargs):
        super().__init__(**kwargs)

        self.info_stream = info_stream
        self.progress = progress
        self.timeout = timeout

    def fetch(self, *args, **kwargs):
        filename, request = self.fetch_file(*args, **kwargs)

        with open(filename, "r") as file_handler:
            data = file_handler.read()

        return data, request

    def fetch_file(self, url, filename=None, params=None, label=None):
        if self.progress:
            info_stream = self.info_stream
        else:
            # null file
            info_stream = open(os.devnull, "w")

        if label is None:
            label = "Downloading"
        label += ": "

        pbar = progressbar.ProgressBar(widgets=[label,
                                                progressbar.Percentage(),
                                                progressbar.Bar(),
                                                progressbar.ETA()],
                                       max_value=100, fd=info_stream).start()

        timeout = self.timeout
        if timeout == 0:
            # avoid invalid timeout
            timeout = None

        # make request
        request = requests.get(url, params=params, stream=True, timeout=timeout)
        total_data_length = int(request.headers.get("content-length"))

        data_length = 0

        if filename is None:
            # create temporary file
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = tmp_file.name

        with open(filename, "wb") as file_handler:
            for chunk in request.iter_content(chunk_size=128):
                if chunk:
                    file_handler.write(chunk)

                    data_length += len(chunk)

                    if data_length == total_data_length:
                        fraction = 100
                    else:
                        fraction = 100 * data_length / total_data_length

                    # check in case lengths are misreported
                    if fraction > 100:
                        fraction = 100
                    elif fraction < 0:
                        fraction = 0

                    pbar.update(fraction)

        pbar.finish()

        return filename, request


def db_to_mag(quantity):
    return 10 ** (quantity / 20)

def mag_to_db(quantity):
    return 20 * np.log10(quantity)
