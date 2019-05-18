"""Data tests"""

from unittest import TestCase
import numpy as np

from zero.components import OpAmp, Node, VoltageNoise
from zero.solution import Solution, matches_between
from zero.data import Series, Response, NoiseDensity, MultiNoiseDensity

# fixed random seed for test reproducibility
np.random.seed(seed=2543070)


class SolutionTestCase(TestCase):
    """Solution tests"""
    def setUp(self):
        # components
        op11 = OpAmp(model="OP00", node1="n11", node2="n12", node3="n13")
        op21 = OpAmp(model="OP00", node1="n21", node2="n22", node3="n23")
        # data points in each set
        count1 = 100
        count2 = 101
        # frequencies
        self.frequencies1 = np.logspace(0, 5, count1)
        self.frequencies2 = np.logspace(0, 5, count2)
        # data
        responsedata11 = np.random.random((count1))
        responsedata12 = np.random.random((count1))
        responsedata13 = np.random.random((count1))
        responsedata21 = np.random.random((count2))
        responsedata22 = np.random.random((count2))
        noisedata11 = np.random.random((count1))
        noisedata12 = np.random.random((count1))
        noisedata21 = np.random.random((count2))
        noisedata22 = np.random.random((count2))
        # sources
        responsesource11 = Node("nso11")
        responsesource21 = Node("ns21")
        noisesource11 = VoltageNoise(component=op11)
        noisesource21 = VoltageNoise(component=op21)
        # sinks
        responsesink11 = Node("ntfs11")
        responsesink12 = Node("ntfs12")
        responsesink13 = Node("ntfs13")
        responsesink21 = Node("ntfs21")
        responsesink22 = Node("ntfs22")
        noisesink11 = Node("nns11")
        noisesink12 = Node("nns12")
        noisesink21 = Node("nns21")
        noisesink22 = Node("nns22")
        # series
        responseseries11 = Series(x=self.frequencies1, y=responsedata11)
        responseseries12 = Series(x=self.frequencies1, y=responsedata12)
        responseseries13 = Series(x=self.frequencies1, y=responsedata13)
        responseseries21 = Series(x=self.frequencies2, y=responsedata21)
        responseseries22 = Series(x=self.frequencies2, y=responsedata22)
        noiseseries11 = Series(x=self.frequencies1, y=noisedata11)
        noiseseries12 = Series(x=self.frequencies1, y=noisedata12)
        noiseseries21 = Series(x=self.frequencies2, y=noisedata21)
        noiseseries22 = Series(x=self.frequencies2, y=noisedata22)
        # set 1 functions
        self.response11 = Response(source=responsesource11, sink=responsesink11,
                                   series=responseseries11)
        self.response12 = Response(source=responsesource11, sink=responsesink12,
                                   series=responseseries12)
        self.response13 = Response(source=responsesource11, sink=responsesink13,
                                   series=responseseries13)
        self.noise11 = NoiseDensity(source=noisesource11, sink=noisesink11, series=noiseseries11)
        self.noise12 = NoiseDensity(source=noisesource11, sink=noisesink12, series=noiseseries12)
        # set 2 functions
        self.response21 = Response(source=responsesource21, sink=responsesink21,
                                   series=responseseries21)
        self.response22 = Response(source=responsesource21, sink=responsesink22,
                                   series=responseseries22)
        self.noise21 = NoiseDensity(source=noisesource21, sink=noisesink21, series=noiseseries21)
        self.noise22 = NoiseDensity(source=noisesource21, sink=noisesink22, series=noiseseries22)

    def test_solutions_equal(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)

        sol_b = Solution(self.frequencies1)
        sol_b.add_response(self.response11)
        sol_b.add_response(self.response12)

        self.assertTrue(sol_a.equivalent_to(sol_b))

    def test_constituent_noise_sum_equal_total_noise_sum(self):
        sources = [VoltageNoise(component=OpAmp(model="OP00", node1="n1", node2="n2", node3="n3")),
                   VoltageNoise(component=OpAmp(model="OP00", node1="n4", node2="n5", node3="n6"))]
        op = OpAmp(model="OP00", node1="n1", node2="n2", node3="n3")
        noise1 = self.noise11
        series2 = Series(x=noise1.frequencies, y=noise1.spectral_density)
        sink = noise1.sink
        noise2 = NoiseDensity(source=VoltageNoise(component=op), sink=sink, series=series2)
        constituents = [noise1, noise2]
        noise_sum = np.sqrt(sum([noise.spectral_density ** 2 for noise in constituents]))
        sum_series = Series(self.frequencies1, noise_sum)

        # sum from constituents
        noisesum1 = MultiNoiseDensity(sink=sink, constituents=constituents)
        # sum from total
        noisesum2 = MultiNoiseDensity(sources=sources, sink=sink, series=sum_series)

        self.assertTrue(noisesum1.equivalent(noisesum2))

    def test_solutions_not_equal_response_frequencies(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)

        sol_b = Solution(self.frequencies2)
        sol_b.add_response(self.response21)
        sol_b.add_response(self.response22)

        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_not_equal_noise_frequencies(self):
        sol_c = Solution(self.frequencies1)
        sol_c.add_noise(self.noise11)
        sol_c.add_noise(self.noise12)

        sol_d = Solution(self.frequencies2)
        sol_d.add_noise(self.noise21)
        sol_d.add_noise(self.noise22)

        self.assertFalse(sol_c.equivalent_to(sol_d))

    def test_solutions_not_equal_responses(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)

        sol_b = Solution(self.frequencies1)
        sol_b.add_response(self.response11)

        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_not_equal_noise(self):
        sol_c = Solution(self.frequencies1)
        sol_c.add_response(self.noise11)
        sol_c.add_response(self.noise12)

        sol_d = Solution(self.frequencies1)
        sol_d.add_response(self.noise11)

        self.assertFalse(sol_c.equivalent_to(sol_d))

    def test_solution_matching(self):
        """Test method to report differences between solutions"""
        # no matches
        sol_a = Solution(self.frequencies1)
        sol_b = Solution(self.frequencies1)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertFalse(matches)
        self.assertFalse(residuals_a)
        self.assertFalse(residuals_b)

        # one shared match, one non-shared in first
        sol_a = Solution(self.frequencies1)
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)
        sol_b = Solution(self.frequencies1)
        sol_b.add_response(self.response11)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertEqual(matches, [(self.response11, self.response11)])
        self.assertEqual(residuals_a, [self.response12])
        self.assertFalse(residuals_b)

        # one shared match, one non-shared in both
        sol_a = Solution(self.frequencies1)
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)
        sol_b = Solution(self.frequencies1)
        sol_b.add_response(self.response11)
        sol_b.add_response(self.response13)
        matches, residuals_a, residuals_b = matches_between(sol_a, sol_b)
        self.assertEqual(matches, [(self.response11, self.response11)])
        self.assertEqual(residuals_a, [self.response12])
        self.assertEqual(residuals_b, [self.response13])

    def test_solution_combination(self):
        """Test method to combine solutions"""
        sol_a = Solution(self.frequencies1)
        sol_a.name = "Sol A"
        sol_a.add_response(self.response11)
        sol_a.add_response(self.response12)

        sol_b = Solution(self.frequencies1)
        sol_b.name = "Sol B"
        sol_b.add_response(self.response13)

        # Combine.
        sol_c = sol_a + sol_b

        self.assertCountEqual(sol_c.groups, ["Sol A", "Sol B"])
        self.assertCountEqual(sol_c.functions["Sol A"], [self.response11, self.response12])
        self.assertCountEqual(sol_c.functions["Sol B"], [self.response13])
