"""Test case with methods for constructing mock Zero data."""

import abc
from unittest import TestCase
import numpy as np

from zero.components import (Resistor, Capacitor, Inductor, OpAmp, Node, OpAmpVoltageNoise,
                             OpAmpCurrentNoise)
from zero.solution import Solution
from zero.data import Series, Response, NoiseDensity, MultiNoiseDensity

# Fixed random seed for test reproducibility.
np.random.seed(seed=2543070)


class ZeroDataTestCase(TestCase, metaclass=abc.ABCMeta):
    """Zero data test case"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_node_num = 0
        self._last_resistor_num = 0
        self._last_capacitor_num = 0
        self._last_inductor_num = 0
        self._last_opamp_num = 0

    def _unique_node_name(self):
        self._last_node_num += 1
        return f"n{self._last_node_num}"

    def _unique_resistor_name(self):
        self._last_resistor_num += 1
        return f"r{self._last_resistor_num}"

    def _unique_capacitor_name(self):
        self._last_capacitor_num += 1
        return f"c{self._last_capacitor_num}"

    def _unique_inductor_name(self):
        self._last_inductor_num += 1
        return f"l{self._last_inductor_num}"

    def _unique_opamp_name(self):
        self._last_opamp_num += 1
        return f"op{self._last_opamp_num}"

    def _data(self, shape, cplx=False):
        data = np.random.random(shape)
        if cplx:
            data = data + 1j * self._data(shape, False)
        return data

    def _freqs(self, n=10):
        return np.sort(self._data(n))

    def _series(self, freqs, data=None, cplx=False):
        if data is None:
            data = self._data(len(freqs), cplx)
        return Series(freqs, data)

    def _node(self):
        return Node(self._unique_node_name())

    def _opamp(self, node1=None, node2=None, node3=None, model=None):
        if node1 is None:
            node1 = self._node()
        if node2 is None:
            node2 = self._node()
        if node3 is None:
            node3 = self._node()
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

    def _capacitor(self, node1=None, node2=None, value=None):
        if node1 is None:
            node1 = self._node()
        if node2 is None:
            node2 = self._node()
        if value is None:
            value = "1u"
        return Capacitor(name=self._unique_capacitor_name(), node1=node1, node2=node2, value=value)

    def _inductor(self, node1=None, node2=None, value=None):
        if node1 is None:
            node1 = self._node()
        if node2 is None:
            node2 = self._node()
        if value is None:
            value = "1u"
        return Inductor(name=self._unique_inductor_name(), node1=node1, node2=node2, value=value)

    def _voltage_noise(self, component=None):
        if component is None:
            component = self._resistor()
        return OpAmpVoltageNoise(component=component)

    def _current_noise(self, node=None, component=None):
        if node is None:
            node = self._node()
        if component is None:
            component = self._resistor(node1=node)
        return OpAmpCurrentNoise(node=node, component=component)

    def _response(self, source, sink, freqs):
        return Response(source=source, sink=sink, series=self._series(freqs, cplx=True))

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

    def _vnoise_at_node(self, freqs, source=None, sink=None):
        if source is None:
            source = self._voltage_noise()
        if sink is None:
            sink = self._node()
        return self._noise_density(freqs, source, sink)

    def _vnoise_at_comp(self, freqs, source=None, sink=None):
        if source is None:
            source = self._voltage_noise()
        if sink is None:
            sink = self._resistor()
        return self._noise_density(freqs, source, sink)

    def _inoise_at_node(self, freqs, source=None, sink=None):
        if source is None:
            source = self._current_noise()
        if sink is None:
            sink = self._node()
        return self._noise_density(freqs, source, sink)

    def _inoise_at_comp(self, freqs, source=None, sink=None):
        if source is None:
            source = self._current_noise()
        if sink is None:
            sink = self._resistor()
        return self._noise_density(freqs, source, sink)

    def _multi_noise_density(self, sink, constituents, label=None):
        return MultiNoiseDensity(sink=sink, constituents=constituents, label=label)

    def _solution(self, freq):
        return Solution(freq)
