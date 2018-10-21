"""Data tests"""

from unittest import TestCase
import numpy as np

from circuit.components import OpAmp, Node, VoltageNoise
from circuit.solution import Solution
from circuit.data import Series, TransferFunction, NoiseSpectrum, MultiNoiseSpectrum

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
        tfdata11 = np.random.random((count1))
        tfdata12 = np.random.random((count1))
        tfdata21 = np.random.random((count2))
        tfdata22 = np.random.random((count2))
        noisedata11 = np.random.random((count1))
        noisedata12 = np.random.random((count1))
        noisedata21 = np.random.random((count2))
        noisedata22 = np.random.random((count2))
        # sources
        tfsource11 = Node("nso11")
        tfsource21 = Node("ns21")
        noisesource11 = VoltageNoise(component=op11)
        noisesource21 = VoltageNoise(component=op21)
        # sinks
        tfsink11 = Node("ntfs11")
        tfsink12 = Node("ntfs12")
        tfsink21 = Node("ntfs21")
        tfsink22 = Node("ntfs22")
        noisesink11 = Node("nns11")
        noisesink12 = Node("nns12")
        noisesink21 = Node("nns21")
        noisesink22 = Node("nns22")
        # series
        tfseries11 = Series(x=self.frequencies1, y=tfdata11)
        tfseries12 = Series(x=self.frequencies1, y=tfdata12)
        tfseries21 = Series(x=self.frequencies2, y=tfdata21)
        tfseries22 = Series(x=self.frequencies2, y=tfdata22)
        noiseseries11 = Series(x=self.frequencies1, y=noisedata11)
        noiseseries12 = Series(x=self.frequencies1, y=noisedata12)
        noiseseries21 = Series(x=self.frequencies2, y=noisedata21)
        noiseseries22 = Series(x=self.frequencies2, y=noisedata22)
        # set 1 functions
        self.tf11 = TransferFunction(source=tfsource11, sink=tfsink11, series=tfseries11)
        self.tf12 = TransferFunction(source=tfsource11, sink=tfsink12, series=tfseries12)
        self.noise11 = NoiseSpectrum(source=noisesource11, sink=noisesink11, series=noiseseries11)
        self.noise12 = NoiseSpectrum(source=noisesource11, sink=noisesink12, series=noiseseries12)
        # set 2 functions
        self.tf21 = TransferFunction(source=tfsource21, sink=tfsink21, series=tfseries21)
        self.tf22 = TransferFunction(source=tfsource21, sink=tfsink22, series=tfseries22)
        self.noise21 = NoiseSpectrum(source=noisesource21, sink=noisesink21, series=noiseseries21)
        self.noise22 = NoiseSpectrum(source=noisesource21, sink=noisesink22, series=noiseseries22)

    def test_solutions_equal(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_tf(self.tf11)
        sol_a.add_tf(self.tf12)

        sol_b = Solution(self.frequencies1)
        sol_b.add_tf(self.tf11)
        sol_b.add_tf(self.tf12)

        self.assertTrue(sol_a.equivalent_to(sol_b))

    def test_constituent_noise_sum_equal_total_noise_sum(self):
        sources = [VoltageNoise(component=OpAmp(model="OP00", node1="n1", node2="n2", node3="n3")),
                   VoltageNoise(component=OpAmp(model="OP00", node1="n4", node2="n5", node3="n6"))]
        op = OpAmp(model="OP00", node1="n1", node2="n2", node3="n3")
        noise1 = self.noise11
        series2 = Series(x=noise1.frequencies, y=noise1.spectrum)
        sink = noise1.sink
        noise2 = NoiseSpectrum(source=VoltageNoise(component=op), sink=sink, series=series2)
        constituents = [noise1, noise2]
        noise_sum = np.sqrt(sum([noise.spectrum ** 2 for noise in constituents]))
        sum_series = Series(self.frequencies1, noise_sum)

        # sum from constituents
        noisesum1 = MultiNoiseSpectrum(sources=sources, sink=sink, constituents=constituents)
        # sum from total
        noisesum2 = MultiNoiseSpectrum(sources=sources, sink=sink, series=sum_series)

        self.assertTrue(noisesum1.equivalent(noisesum2))

    def test_solutions_not_equal_tf_frequencies(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_tf(self.tf11)
        sol_a.add_tf(self.tf12)

        sol_b = Solution(self.frequencies2)
        sol_b.add_tf(self.tf21)
        sol_b.add_tf(self.tf22)

        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_not_equal_noise_frequencies(self):
        sol_c = Solution(self.frequencies1)
        sol_c.add_noise(self.noise11)
        sol_c.add_noise(self.noise12)

        sol_d = Solution(self.frequencies2)
        sol_d.add_noise(self.noise21)
        sol_d.add_noise(self.noise22)

        self.assertFalse(sol_c.equivalent_to(sol_d))

    def test_solutions_not_equal_tfs(self):
        sol_a = Solution(self.frequencies1)
        sol_a.add_tf(self.tf11)
        sol_a.add_tf(self.tf12)

        sol_b = Solution(self.frequencies1)
        sol_b.add_tf(self.tf11)

        self.assertFalse(sol_a.equivalent_to(sol_b))

    def test_solutions_not_equal_noise(self):
        sol_c = Solution(self.frequencies1)
        sol_c.add_tf(self.noise11)
        sol_c.add_tf(self.noise12)

        sol_d = Solution(self.frequencies1)
        sol_d.add_tf(self.noise11)

        self.assertFalse(sol_c.equivalent_to(sol_d))
