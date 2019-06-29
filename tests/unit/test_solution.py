"""Data tests"""

from unittest import TestCase
import numpy as np

from zero.components import OpAmp, Resistor, Node, VoltageNoise, CurrentNoise
from zero.solution import Solution, matches_between
from zero.data import Series, Response, NoiseDensity, MultiNoiseDensity

# fixed random seed for test reproducibility
np.random.seed(seed=2543070)


class BaseSolutionTestCase(TestCase):
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

    def _data(self, n):
        return np.random.random((n))

    def _freqs(self, n=10, start=0, stop=5):
        return np.logspace(start, stop, n)

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


class SolutionFunctionTestCase(BaseSolutionTestCase):
    """Solution function tests"""
    def test_solutions_with_identical_responses_equal(self):
        f = self._freqs()
        resp1 = self._v_v_response(f)
        resp2 = self._v_i_response(f)
        sol_a = self._solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        sol_b.add_response(resp2)
        self.assertTrue(sol_a.equivalent_to(sol_b))

    def test_solutions_with_identical_noise_equal(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
        sol_a = self._solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2)
        sol_b = Solution(f)
        sol_b.add_noise(noise1)
        sol_b.add_noise(noise2)
        self.assertTrue(sol_a.equivalent_to(sol_b))

    def test_solutions_with_different_response_frequencies_not_equal(self):
        f1 = self._freqs()
        sol_a = self._solution(f1)
        sol_a.add_response(self._v_v_response(f1))
        sol_a.add_response(self._v_i_response(f1))
        f2 = self._freqs()
        sol_b = Solution(f2)
        sol_b.add_response(self._v_v_response(f2))
        sol_b.add_response(self._v_i_response(f2))
        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_with_different_noise_frequencies_not_equal(self):
        f1 = self._freqs()
        sol_a = self._solution(f1)
        sol_a.add_noise(self._voltage_noise_at_node(f1))
        sol_a.add_noise(self._voltage_noise_at_node(f1))
        f2 = self._freqs()
        sol_b = Solution(f2)
        sol_b.add_noise(self._voltage_noise_at_node(f2))
        sol_b.add_noise(self._voltage_noise_at_node(f2))
        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_constituent_noise_sum_equal_total_noise_sum(self):
        # TODO: move to data tests.
        f = self._freqs()
        sink = self._resistor()
        noise1 = self._voltage_noise_at_comp(f, sink=sink)
        noise2 = self._voltage_noise_at_comp(f, sink=sink) # Share sink.
        constituents = [noise1, noise2]
        sum_data = np.sqrt(sum([noise.spectral_density ** 2 for noise in constituents]))
        sum_series = self._series(f, sum_data)
        noisesum1 = MultiNoiseDensity(sink=sink, constituents=constituents)
        noisesum2 = MultiNoiseDensity(sources=[noise1.source, noise2.source],
                                      sink=sink, series=sum_series)
        self.assertTrue(noisesum1.equivalent(noisesum2))

    def test_solutions_with_same_freqs_different_responses_not_equal(self):
        f = self._freqs()
        # All responses different.
        sol_a = Solution(f)
        sol_a.add_response(self._v_v_response(f))
        sol_a.add_response(self._v_i_response(f))
        sol_b = Solution(f)
        sol_b.add_response(self._v_v_response(f))
        sol_b.add_response(self._v_i_response(f))
        self.assertFalse(sol_a.equivalent_to(sol_b))
        # One response same, but extra in one solution.
        resp1 = self._v_v_response(f)
        sol_c = Solution(f)
        sol_c.add_response(resp1)
        sol_d = Solution(f)
        sol_d.add_response(resp1)
        sol_d.add_response(self._v_i_response(f))
        self.assertFalse(sol_c.equivalent_to(sol_d))

    def test_solutions_with_same_freqs_different_noise_not_equal(self):
        f = self._freqs()
        # All responses different.
        sol_a = Solution(f)
        sol_a.add_noise(self._voltage_noise_at_node(f))
        sol_a.add_noise(self._voltage_noise_at_node(f))
        sol_b = Solution(f)
        sol_b.add_noise(self._voltage_noise_at_node(f))
        sol_b.add_noise(self._voltage_noise_at_node(f))
        self.assertFalse(sol_a.equivalent_to(sol_b))
        # One noise same, but extra in one solution.
        spectrum1 = self._voltage_noise_at_node(f)
        sol_c = Solution(f)
        sol_c.add_noise(spectrum1)
        sol_d = Solution(f)
        sol_d.add_noise(spectrum1)
        sol_d.add_noise(self._voltage_noise_at_node(f))
        self.assertFalse(sol_c.equivalent_to(sol_d))

    def test_solutions_with_different_freq_responses_not_equal(self):
        f1 = self._freqs()
        f2 = self._freqs()
        # Responses with identical sources and sinks but different frequencies.
        resp1a = self._v_v_response(f1)
        resp1b = self._v_v_response(f2, node_source=resp1a.source, node_sink=resp1a.sink)
        sol_a = Solution(f1)
        sol_a.add_response(resp1a)
        sol_b = Solution(f2)
        sol_b.add_response(resp1b)
        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_with_different_freq_noise_not_equal(self):
        f1 = self._freqs()
        f2 = self._freqs()
        # Noise with identical sources and sinks but different frequencies.
        spectrum1a = self._voltage_noise_at_node(f1)
        spectrum1b = self._voltage_noise_at_node(f2, source=spectrum1a.source, sink=spectrum1a.sink)
        sol_a = Solution(f1)
        sol_a.add_noise(spectrum1a)
        sol_b = Solution(f2)
        sol_b.add_noise(spectrum1b)
        self.assertFalse(sol_a.equivalent_to(sol_b))


class SolutionCombinationTestCase(BaseSolutionTestCase):
    """Solution combination tests"""
    def test_solution_matching_no_matches(self):
        # No matches.
        f = self._freqs()
        sol_a = Solution(f)
        sol_b = Solution(f)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertFalse(matches)
        self.assertFalse(residuals_a)
        self.assertFalse(residuals_b)

    def test_solution_matching_one_non_shared_match(self):
        f = self._freqs()
        resp1 = self._v_i_response(f)
        resp2 = self._v_i_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertEqual(matches, [(resp1, resp1)])
        self.assertEqual(residuals_a, [resp2])
        self.assertFalse(residuals_b)

    def test_solution_matching_one_non_shared_match_in_each(self):
        f = self._freqs()
        resp1 = self._v_i_response(f)
        resp2 = self._v_i_response(f)
        resp3 = self._v_i_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        sol_b.add_response(resp3)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertEqual(matches, [(resp1, resp1)])
        self.assertEqual(residuals_a, [resp2])
        self.assertEqual(residuals_b, [resp3])

    def test_solution_combination(self):
        """Test method to combine solutions"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.name = "Sol A"
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.name = "Sol B"
        sol_b.add_response(resp3)

        # Combine.
        sol_c = sol_a + sol_b

        # Check groups.
        self.assertCountEqual(sol_c.groups, ["Sol A", "Sol B", sol_c.DEFAULT_REF_GROUP_NAME])
        # Check functions.
        self.assertCountEqual(sol_c.functions["Sol A"], [resp1, resp2])
        self.assertCountEqual(sol_c.functions["Sol B"], [resp3])


class SolutionFilterTestCase(BaseSolutionTestCase):
    """Solution filter tests"""
    def test_get_response_no_group(self):
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol = Solution(f)
        sol.add_response(resp1)
        sol.add_response(resp2)
        self.assertEqual(sol.get_response(source=resp1.source, sink=resp1.sink), resp1)
        self.assertEqual(sol.get_response(source=resp2.source, sink=resp2.sink), resp2)

    def test_get_response_with_group(self):
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol = Solution(f)
        sol.add_response(resp1, group="b")
        sol.add_response(resp2, group="b")
        self.assertEqual(sol.get_response(source=resp1.source, sink=resp1.sink, group="b"), resp1)
        self.assertEqual(sol.get_response(source=resp2.source, sink=resp2.sink, group="b"), resp2)
        # Default groups shouldn't have the response.
        self.assertRaises(ValueError, sol.get_response, source=resp1.source, sink=resp1.sink)
        self.assertRaises(ValueError, sol.get_response, source=resp2.source, sink=resp2.sink)

    def test_get_response_with_label(self):
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol = Solution(f)
        sol.add_response(resp1)
        sol.add_response(resp2)
        label1 = f"{resp1.source.name} to {resp1.sink.name} (A/A)"
        label2 = f"{resp2.source.name} to {resp2.sink.name} (V/A)"
        self.assertEqual(sol.get_response(label=label1), resp1)
        self.assertEqual(sol.get_response(label=label2), resp2)
        # Without units.
        self.assertRaises(ValueError, sol.get_response, label=label1[:-6])

    def test_get_noise_no_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertEqual(sol.get_noise(source=noise1.source, sink=noise1.sink), noise1)
        self.assertEqual(sol.get_noise(source=noise2.source, sink=noise2.sink), noise2)

    def test_get_noise_with_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f)
        sol = Solution(f)
        sol.add_noise(noise1, group="b")
        sol.add_noise(noise2, group="b")
        self.assertEqual(sol.get_noise(source=noise1.source, sink=noise1.sink, group="b"), noise1)
        self.assertEqual(sol.get_noise(source=noise2.source, sink=noise2.sink, group="b"), noise2)
        # Default groups shouldn't have the response.
        self.assertRaises(ValueError, sol.get_noise, source=noise1.source, sink=noise1.sink)
        self.assertRaises(ValueError, sol.get_noise, source=noise2.source, sink=noise2.sink)

    def test_get_noise_with_label(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        label1 = f"{noise1.source} to {noise1.sink.name}"
        label2 = f"{noise2.source} to {noise2.sink.name}"
        self.assertEqual(sol.get_noise(label=label1), noise1)
        self.assertEqual(sol.get_noise(label=label2), noise2)

    def test_get_noise_sum_no_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f)
        noise4 = self._current_noise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = Solution(f)
        sol.add_noise_sum(sum1)
        sol.add_noise_sum(sum2)
        self.assertEqual(sol.get_noise_sum(sink=sum1.sink), sum1)
        self.assertEqual(sol.get_noise_sum(sink=sum2.sink), sum2)

    def test_get_noise_sum_with_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f)
        noise4 = self._current_noise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = Solution(f)
        sol.add_noise_sum(sum1, group="b")
        sol.add_noise_sum(sum2, group="b")
        self.assertEqual(sol.get_noise_sum(sink=sum1.sink, group="b"), sum1)
        self.assertEqual(sol.get_noise_sum(sink=sum2.sink, group="b"), sum2)
        # Default groups shouldn't have the response.
        self.assertRaises(ValueError, sol.get_noise_sum, sink=sum1.sink)
        self.assertRaises(ValueError, sol.get_noise_sum, sink=sum2.sink)

    def test_get_noise_sum_with_label(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f)
        noise4 = self._current_noise_at_comp(f, sink=noise3.sink)
        label1 = "label 1"
        label2 = "label 2"
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2], label=label1)
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4], label=label2)
        sol = Solution(f)
        sol.add_noise_sum(sum1)
        sol.add_noise_sum(sum2)
        self.assertEqual(sol.get_noise_sum(label=label1), sum1)
        self.assertEqual(sol.get_noise_sum(label=label2), sum2)
