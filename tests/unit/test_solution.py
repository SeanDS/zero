"""Solution unit tests"""

import numpy as np
from zero.solution import Solution, matches_between
from ..data import ZeroDataTestCase


class SolutionFilterTestCase(ZeroDataTestCase):
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

    def test_get_response_with_degenerate_functions_same_source(self):
        f = self._freqs()
        res1 = self._resistor()
        resp1 = self._i_i_response(f, component_source=res1)
        resp2 = self._i_v_response(f, component_source=res1)
        sol = Solution(f)
        sol.add_response(resp1)
        sol.add_response(resp2)
        self.assertRaises(ValueError, sol.get_response, source=res1)

    def test_get_response_with_degenerate_functions_same_sink(self):
        f = self._freqs()
        res1 = self._resistor()
        resp1 = self._i_i_response(f, component_sink=res1)
        resp2 = self._v_i_response(f, component_sink=res1)
        sol = Solution(f)
        sol.add_response(resp1)
        sol.add_response(resp2)
        self.assertRaises(ValueError, sol.get_response, sink=res1)

    def test_get_noise_no_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertEqual(sol.get_noise(source=noise1.source, sink=noise1.sink), noise1)
        self.assertEqual(sol.get_noise(source=noise2.source, sink=noise2.sink), noise2)

    def test_get_noise_with_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f)
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
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        label1 = f"{noise1.source} to {noise1.sink.name}"
        label2 = f"{noise2.source} to {noise2.sink.name}"
        self.assertEqual(sol.get_noise(label=label1), noise1)
        self.assertEqual(sol.get_noise(label=label2), noise2)

    def test_get_noise_with_degenerate_functions_same_source(self):
        f = self._freqs()
        res1 = self._resistor()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._vnoise_at_comp(f, source=noise1.source)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertRaises(ValueError, sol.get_noise, source=noise1.source)

    def test_get_noise_with_degenerate_functions_same_sink(self):
        f = self._freqs()
        res1 = self._resistor()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._vnoise_at_comp(f, sink=noise1.sink)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertRaises(ValueError, sol.get_noise, sink=noise1.sink)

    def test_get_noise_sum_no_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f)
        noise4 = self._inoise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = Solution(f)
        sol.add_noise_sum(sum1)
        sol.add_noise_sum(sum2)
        self.assertEqual(sol.get_noise_sum(sink=sum1.sink), sum1)
        self.assertEqual(sol.get_noise_sum(sink=sum2.sink), sum2)

    def test_get_noise_sum_with_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f)
        noise4 = self._inoise_at_comp(f, sink=noise3.sink)
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
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f)
        noise4 = self._inoise_at_comp(f, sink=noise3.sink)
        label1 = "label 1"
        label2 = "label 2"
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2], label=label1)
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4], label=label2)
        sol = Solution(f)
        sol.add_noise_sum(sum1)
        sol.add_noise_sum(sum2)
        self.assertEqual(sol.get_noise_sum(label=label1), sum1)
        self.assertEqual(sol.get_noise_sum(label=label2), sum2)

    def test_get_noise_sum_with_degenerate_functions_same_sink(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._vnoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f, sink=noise1.sink)
        noise4 = self._vnoise_at_comp(f, sink=noise1.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise1.sink, [noise3, noise4])
        sol = Solution(f)
        sol.add_noise_sum(sum1)
        sol.add_noise_sum(sum2)
        self.assertRaises(ValueError, sol.get_noise_sum, sink=sum1.sink)


class SolutionFunctionReplacementTestCase(ZeroDataTestCase):
    """Solution function replacement tests"""
    def test_response_replacement_no_group(self):
        f = self._freqs()
        resp1 = self._v_v_response(f)
        resp2 = self._v_i_response(f)
        sol = self._solution(f)
        sol.add_response(resp1)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [resp1])
        sol.replace(resp1, resp2)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [resp2])

    def test_response_replacement_with_group(self):
        f = self._freqs()
        resp1 = self._v_v_response(f)
        resp2 = self._v_i_response(f)
        sol = self._solution(f)
        sol.add_response(resp1, group="b")
        self.assertEqual(sol.functions["b"], [resp1])
        # Trying to replace without specifying group is not allowed.
        self.assertRaises(ValueError, sol.replace, resp1, resp2)
        sol.replace(resp1, resp2, group="b")
        self.assertEqual(sol.functions["b"], [resp2])

    def test_noise_replacement_no_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f)
        sol = self._solution(f)
        sol.add_noise(noise1)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [noise1])
        sol.replace(noise1, noise2)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [noise2])

    def test_noise_replacement_with_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f)
        sol = self._solution(f)
        sol.add_noise(noise1, group="b")
        self.assertEqual(sol.functions["b"], [noise1])
        # Trying to replace without specifying group is not allowed.
        self.assertRaises(ValueError, sol.replace, noise1, noise2)
        sol.replace(noise1, noise2, group="b")
        self.assertEqual(sol.functions["b"], [noise2])

    def test_noise_sum_replacement_no_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f)
        noise4 = self._inoise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = self._solution(f)
        sol.add_noise_sum(sum1)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [sum1])
        sol.replace(sum1, sum2)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [sum2])

    def test_noise_sum_replacement_with_group(self):
        f = self._freqs()
        noise1 = self._vnoise_at_comp(f)
        noise2 = self._inoise_at_comp(f, sink=noise1.sink)
        noise3 = self._vnoise_at_comp(f)
        noise4 = self._inoise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = self._solution(f)
        sol.add_noise_sum(sum1, group="b")
        self.assertEqual(sol.functions["b"], [sum1])
        # Trying to replace without specifying group is not allowed.
        self.assertRaises(ValueError, sol.replace, sum1, sum2)
        sol.replace(sum1, sum2, group="b")
        self.assertEqual(sol.functions["b"], [sum2])


class SolutionScalingTestCase(ZeroDataTestCase):
    """Solution function scaling tests"""
    def test_response_scaling(self):
        f = self._freqs()
        node1 = self._node()
        node2 = self._node()
        label1 = "resp1"
        label2 = "resp2"
        label3 = "resp3"
        resp1 = self._v_v_response(f, node_sink=node1)
        resp1.label = label1
        resp2 = self._v_v_response(f, node_sink=node1)
        resp2.label = label2
        resp3 = self._v_v_response(f, node_source=node1, node_sink=node2)
        resp3.label = label3
        sol = self._solution(f)
        sol.add_response(resp1)
        sol.add_response(resp2)
        solresp1a = sol.get_response(label=label1)
        solresp2a = sol.get_response(label=label2)
        self.assertEqual(solresp1a, resp1)
        self.assertEqual(solresp2a, resp2)
        sol.scale_responses(resp3)
        solresp1b = sol.get_response(label=label1)
        solresp2b = sol.get_response(label=label2)
        self.assertEqual(solresp1b.sink, resp3.sink)
        self.assertEqual(solresp2b.sink, resp3.sink)

    def test_noise_scaling(self):
        f = self._freqs()
        node1 = self._node()
        node2 = self._node()
        label1 = "noise1"
        label2 = "noise2"
        label3 = "resp1"
        noise1 = self._vnoise_at_node(f, sink=node1)
        noise1.label = label1
        noise2 = self._vnoise_at_node(f, sink=node1)
        noise2.label = label2
        resp = self._v_v_response(f, node_source=node1, node_sink=node2)
        resp.label = label3
        sol = self._solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        solnoise1a = sol.get_noise(label=label1)
        solnoise2a = sol.get_noise(label=label2)
        self.assertEqual(solnoise1a, noise1)
        self.assertEqual(solnoise2a, noise2)
        sol.scale_noise(resp)
        solnoise1b = sol.get_noise(label=label1)
        solnoise2b = sol.get_noise(label=label2)
        self.assertEqual(solnoise1b.sink, resp.sink)
        self.assertEqual(solnoise2b.sink, resp.sink)
