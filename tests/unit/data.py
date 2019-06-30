"""Test case with methods for constructing mock data."""

import abc
from unittest import TestCase
import numpy as np

from zero.components import OpAmp, Resistor, Node, VoltageNoise, CurrentNoise
from zero.solution import Solution
from zero.data import Series, Response, NoiseDensity, MultiNoiseDensity

# fixed random seed for test reproducibility
np.random.seed(seed=2543070)


class ZeroDataTestCase(TestCase, metaclass=abc.ABCMeta):
    """Zero data test case"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_node_num = 0
        self._last_resistor_num = 0
        self._last_opamp_num = 0

    def _unique_node_name(self):
        self._last_node_num += 1
        return f"n{self._last_node_num}"

    def _unique_resistor_name(self):
        self._last_resistor_num += 1
        return f"r{self._last_resistor_num}"

    def _unique_opamp_name(self):
        self._last_opamp_num += 1
        return f"op{self._last_opamp_num}"

    def _data(self, shape):
        return np.random.random(shape)

    def _freqs(self, n=10):
        return np.sort(self._data(n))

    def _series(self, freqs, data=None):
        if data is None:
            data = self._data(len(freqs))
        return Series(freqs, data)

    def _node(self):
        return Node(self._unique_node_name())

    def _opamp(self, node1, node2, node3, model=None):
        if model is None:
            model = "OP00"
        return OpAmp(name=self._unique_opamp_name(), model=model, node1=node1, node2=node2,
                     node3=node3)

    def _resistor(self, node1=None, node2=None, value=None):
        if node1 is None:
            node1 = self._node()
        if node2 is None:
            node2 = self._node()
        if value is None:
            value = "1k"
        return Resistor(name=self._unique_resistor_name(), node1=node1, node2=node2, value=value)

    def _voltage_noise(self, component=None):
        if component is None:
            component = self._resistor()
        return VoltageNoise(component)

    def _current_noise(self, node=None, component=None):
        if node is None:
            node = self._node()
        if component is None:
            component = self._resistor(node1=node)
        return CurrentNoise(node, component)

    def _response(self, source, sink, freqs):
        return Response(source=source, sink=sink, series=self._series(freqs))

    def _v_v_response(self, freqs, node_source=None, node_sink=None):
        if node_source is None:
            node_source = self._node()
        if node_sink is None:
            node_sink = self._node()
        return self._response(node_source, node_sink, freqs)

    def _v_i_response(self, freqs, node_source=None, component_sink=None):
        if node_source is None:
            node_source = self._node()
        if component_sink is None:
            component_sink = self._resistor()
        return self._response(node_source, component_sink, freqs)

    def _i_i_response(self, freqs, component_source=None, component_sink=None):
        if component_source is None:
            component_source = self._resistor()
        if component_sink is None:
            component_sink = self._resistor()
        return self._response(component_source, component_sink, freqs)

    def _i_v_response(self, freqs, component_source=None, node_sink=None):
        if component_source is None:
            component_source = self._resistor()
        if node_sink is None:
            node_sink = self._node()
        return self._response(component_source, node_sink, freqs)

    def _noise_density(self, freqs, source, sink):
        return NoiseDensity(source=source, sink=sink, series=self._series(freqs))

    def _voltage_noise_at_node(self, freqs, source=None, sink=None):
        if source is None:
            source = self._voltage_noise()
        if sink is None:
            sink = self._node()
        return self._noise_density(freqs, source, sink)

    def _voltage_noise_at_comp(self, freqs, source=None, sink=None):
        if source is None:
            source = self._voltage_noise()
        if sink is None:
            sink = self._resistor()
        return self._noise_density(freqs, source, sink)

    def _current_noise_at_node(self, freqs, source=None, sink=None):
        if source is None:
            source = self._current_noise()
        if sink is None:
            sink = self._node()
        return self._noise_density(freqs, source, sink)

    def _current_noise_at_comp(self, freqs, source=None, sink=None):
        if source is None:
            source = self._current_noise()
        if sink is None:
            sink = self._resistor()
        return self._noise_density(freqs, source, sink)

    def _multi_noise_density(self, sink, constituents, label=None):
        return MultiNoiseDensity(sink=sink, constituents=constituents, label=label)

    def _solution(self, freq):
        return Solution(freq)
