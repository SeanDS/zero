"""Component library parser tests"""

import os
import json
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as np_assert_array_almost_equal

from zero.config import OpAmpLibrary


class NullOpAmpLibrary(OpAmpLibrary):
    """Op-amp library which loads nothing by default, to facilitate incremental testing"""

    @property
    def base_config_path(self):
        """Overridden base config path"""
        return os.devnull

    @property
    def user_config_path(self):
        """Overridden user config path"""
        return os.devnull

    def parse_opamp_test_data(self, name, data):
        """Wrapper to parse an op-amp as if it came from the YAML file"""
        return self._parse_lib_data(name, data)

    def find_library_opamp_by_test_name(self, name):
        name = self.format_name(name)

        for opamp in self.opamps:
            if opamp.model == name:
                return opamp

        raise ValueError(f"op-amp '{name}' not found")


class OpAmpPolesTestCase(unittest.TestCase):
    def setUp(self):
        self.reset()

    def reset(self):
        """Reset library"""
        self.library = NullOpAmpLibrary()

    def _parse_and_return(self, data):
        """Parse the specified op-amp data and return the corresponding library op-amp"""
        # Generate unique name.
        # Slow JSON serialisation used to avoid implementing our own hashing method:
        #   https://stackoverflow.com/questions/5884066/hashing-a-dictionary
        unique_hash = hash(json.dumps(data))
        name = f"__testop__{unique_hash}__"

        # parse
        self.library.parse_opamp_test_data(name, data)

        # get parsed model
        return self.library.find_library_opamp_by_test_name(name)

    def test_single_real_poles(self):
        """Test op-amp with single real pole and zero is parsed properly"""
        model = self._parse_and_return({"poles": ["53.4M"], "zeros": ["76.5M"]})

        np_assert_array_almost_equal(np.array([53.4e6]), model.poles)
        np_assert_array_almost_equal(np.array([76.5e6]), model.zeros)

    def test_single_complex_poles(self):
        """Test op-amp with single complex pole and zero is parsed properly"""
        model = self._parse_and_return({"poles": ["53.4M 5.1"], "zeros": ["76.5M 4.7"]})

        np_assert_array_almost_equal(np.array([5235294.117647 - 53142748.287059j,
                                               5235294.117647 + 53142748.287059j]),
                                     model.poles)

        np_assert_array_almost_equal(np.array([8138297.87234042 - 76065880.04973753j,
                                               8138297.87234042 + 76065880.04973753j]),
                                     model.zeros)


    def test_multiple_poles(self):
        """Test op-amp with multiple complex poles and zeros is parsed properly"""
        model = self._parse_and_return({"poles": ["13M", "53.4M 5.1"],
                                        "zeros": ["19.3M", "76.5M 4.7"]})

        np_assert_array_almost_equal(np.array([13e6,
                                               5235294.117647 - 53142748.287059j,
                                               5235294.117647 + 53142748.287059j]),
                                     model.poles)

        np_assert_array_almost_equal(np.array([19.3e6,
                                               8138297.87234042 - 76065880.04973753j,
                                               8138297.87234042 + 76065880.04973753j]),
                                     model.zeros)
