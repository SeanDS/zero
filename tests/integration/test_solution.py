"""Solution integration tests"""

import numpy as np
from zero.solution import Solution, matches_between
from ..data import ZeroDataTestCase


class SolutionEquivalencyTestCase(ZeroDataTestCase):
    """Solution equivalency tests"""
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
        noise1 = self._vnoise_at_node(f)
        noise2 = self._vnoise_at_node(f)
        sol_a = self._solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2)
        sol_b = Solution(f)
        sol_b.add_noise(noise1)
        sol_b.add_noise(noise2)
        self.assertTrue(sol_a.equivalent_to(sol_b))

    def test_solutions_with_different_frequencies_not_equal(self):
        f1 = self._freqs()
        sol_a = self._solution(f1)
        f2 = self._freqs()
        sol_b = Solution(f2)
        self.assertFalse(sol_a.equivalent_to(sol_b))

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
        sol_a.add_noise(self._vnoise_at_node(f))
        sol_a.add_noise(self._vnoise_at_node(f))
        sol_b = Solution(f)
        sol_b.add_noise(self._vnoise_at_node(f))
        sol_b.add_noise(self._vnoise_at_node(f))
        self.assertFalse(sol_a.equivalent_to(sol_b))
        # One noise same, but extra in one solution.
        spectrum1 = self._vnoise_at_node(f)
        sol_c = Solution(f)
        sol_c.add_noise(spectrum1)
        sol_d = Solution(f)
        sol_d.add_noise(spectrum1)
        sol_d.add_noise(self._vnoise_at_node(f))
        self.assertFalse(sol_c.equivalent_to(sol_d))


class SolutionEqualityAndCombinationTestCase(ZeroDataTestCase):
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

    def test_solution_combination_operator(self):
        """Solution combination __mul__ operator should be equivalent to .combine"""
        # Build non-trivial solutions.
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        resp4 = self._v_i_response(f)
        resp5 = self._i_i_response(f)
        resp6 = self._i_v_response(f)
        resp7 = self._v_v_response(f)
        resp8 = self._v_i_response(f)
        resp9 = self._i_i_response(f)
        resp10 = self._i_v_response(f)
        resp11 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.name = "A"
        sol_a.add_response(resp1)
        sol_a.add_response(resp2, group="a")
        sol_a.add_response(resp3, group="b")
        sol_b = Solution(f)
        sol_b.name = "B"
        sol_b.add_response(resp4)
        sol_b.add_response(resp5, group="a")
        sol_b.add_response(resp6, group="c")
        sol_c = Solution(f)
        sol_c.name = "C"
        sol_c.add_response(resp7)
        sol_c.add_response(resp8, group="a")
        sol_c.add_response(resp9, group="b")
        sol_c.add_response(resp10, group="c")
        sol_c.add_response(resp11, group="d")
        # Test combination all at once.
        sol_d_a = sol_a + sol_b + sol_c
        sol_d_b = sol_a.combine(sol_b, sol_c)
        self.assertTrue(sol_d_a.equivalent_to(sol_d_b))
        # Test combination with an intermediate step.
        sol_d_a = sol_a + sol_b
        sol_d_a += sol_c
        sol_d_b = sol_a.combine(sol_b)
        sol_d_b = sol_d_b.combine(sol_c)
        self.assertTrue(sol_d_a.equivalent_to(sol_d_b))
        # Test combination with two intermediate steps.
        sol_d_a = sol_a
        sol_d_a += sol_b
        sol_d_a += sol_c
        sol_d_b = sol_a.combine(sol_b)
        sol_d_b = sol_d_b.combine(sol_c)
        self.assertTrue(sol_d_a.equivalent_to(sol_d_b))

    def test_solution_combination_generated_name(self):
        """Test generated name in combined solution"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.name = "Sol A"
        sol_a.add_response(resp1)
        sol_b = Solution(f)
        sol_b.name = "Sol B"
        sol_b.add_response(resp2)
        sol_c = Solution(f)
        sol_c.name = "Sol C"
        sol_c.add_response(resp3)
        name = f"{sol_a.name} + {sol_b.name} + {sol_c.name}"
        sol_d = sol_a.combine(sol_b, sol_c)
        self.assertEqual(sol_d.name, name)

    def test_solution_combination_specified_name(self):
        """Test specified name in combined solution"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.name = "Sol A"
        sol_a.add_response(resp1)
        sol_b = Solution(f)
        sol_b.name = "Sol B"
        sol_b.add_response(resp2)
        sol_c = Solution(f)
        sol_c.name = "Sol C"
        sol_c.add_response(resp3)
        name = "test"
        sol_d = sol_a.combine(sol_b, sol_c, name=name)
        self.assertEqual(sol_d.name, name)

    def test_solution_combination_different_frequencies(self):
        """Test that combining solutions with different frequency vectors throws error."""
        f1 = self._freqs()
        f2 = self._freqs()
        sol_a = Solution(f1)
        sol_b = Solution(f2)
        self.assertRaises(ValueError, sol_a.combine, sol_b)

    def test_solution_combination_default_groups(self):
        """Test default groups in combined solution"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_b = Solution(f)
        sol_b.add_response(resp2)
        sol_c = Solution(f)
        sol_c.add_response(resp3)
        sol_d = sol_a.combine(sol_b, sol_c)
        # Check functions.
        self.assertCountEqual(sol_d.functions[sol_d.DEFAULT_GROUP_NAME], [resp1, resp2, resp3])

    def test_solution_combination_mixed_groups(self):
        """Test mixed groups in combined solution"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        resp4 = self._v_i_response(f)
        resp5 = self._i_i_response(f)
        resp6 = self._i_v_response(f)
        resp7 = self._v_v_response(f)
        resp8 = self._v_i_response(f)
        resp9 = self._i_i_response(f)
        resp10 = self._i_v_response(f)
        resp11 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2, group="a")
        sol_a.add_response(resp3, group="b")
        sol_b = Solution(f)
        sol_b.add_response(resp4)
        sol_b.add_response(resp5, group="a")
        sol_b.add_response(resp6, group="c")
        sol_c = Solution(f)
        sol_c.add_response(resp7)
        sol_c.add_response(resp8, group="a")
        sol_c.add_response(resp9, group="b")
        sol_c.add_response(resp10, group="c")
        sol_c.add_response(resp11, group="d")
        sol_d = sol_a.combine(sol_b, sol_c)
        # Check functions.
        self.assertCountEqual(sol_d.functions[sol_d.DEFAULT_GROUP_NAME], [resp1, resp4, resp7])
        self.assertCountEqual(sol_d.functions["a"], [resp2, resp5, resp8])
        self.assertCountEqual(sol_d.functions["b"], [resp3, resp9])
        self.assertCountEqual(sol_d.functions["c"], [resp6, resp10])
        self.assertCountEqual(sol_d.functions["d"], [resp11])

    def test_solution_combination_identical_responses_same_group_invalid(self):
        """Test that combining solutions with identical responses in same groups is invalid."""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2, group="b")
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        sol_b.add_response(resp2, group="b")
        self.assertRaises(ValueError, sol_a.combine, sol_b)

    def test_solution_combination_identical_responses_different_group_valid(self):
        """Test that combining solutions with identical responses in different groups is valid."""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.add_response(resp1, group="b")
        sol_b.add_response(resp2, group="b")
        sol_c = sol_a.combine(sol_b)
        self.assertCountEqual(sol_c.functions[sol_c.DEFAULT_GROUP_NAME], [resp1, resp2])
        self.assertCountEqual(sol_c.functions["b"], [resp1, resp2])

    def test_solution_combination_identical_noise_same_group_invalid(self):
        """Test that combining solutions with identical noise in same target groups is invalid."""
        f = self._freqs()
        noise1 = self._vnoise_at_node(f)
        noise2 = self._vnoise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2, group="b")
        sol_b = Solution(f)
        sol_b.add_noise(noise1)
        sol_b.add_noise(noise2, group="b")
        self.assertRaises(ValueError, sol_a.combine, sol_b)

    def test_solution_combination_identical_noise_different_group_valid(self):
        """Test that combining solutions with identical noise in different groups is valid."""
        f = self._freqs()
        noise1 = self._vnoise_at_node(f)
        noise2 = self._vnoise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2)
        sol_b = Solution(f)
        sol_b.add_noise(noise1, group="b")
        sol_b.add_noise(noise2, group="b")
        sol_c = sol_a.combine(sol_b)
        self.assertCountEqual(sol_c.functions[sol_c.DEFAULT_GROUP_NAME], [noise1, noise2])
        self.assertCountEqual(sol_c.functions["b"], [noise1, noise2])

    def test_solution_combination_with_references(self):
        """Test reference curves in combined solution"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        resp3 = self._v_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response_reference(f, np.ones_like(f), label="A")
        sol_a.add_response_reference(f, np.ones_like(f), label="B")
        sol_a.add_noise_reference(f, np.ones_like(f), label="C")
        sol_b = Solution(f)
        sol_b.add_response(resp2)
        sol_b.add_response_reference(f, np.ones_like(f), label="D")
        sol_b.add_noise_reference(f, np.ones_like(f), label="E")
        sol_b.add_noise_reference(f, np.ones_like(f), label="F")
        sol_c = Solution(f)
        sol_c.add_response(resp3)
        sol_c.add_noise_reference(f, np.ones_like(f), label="G")
        sol_c.add_response_reference(f, np.ones_like(f), label="H")
        sol_c.add_noise_reference(f, np.ones_like(f), label="I")
        sol_d = sol_a + sol_b + sol_c
        self.assertCountEqual([ref.label for ref in sol_d.response_references],
                              ["A", "B", "D", "H"])
        self.assertCountEqual([ref.label for ref in sol_d.noise_references],
                              ["C", "E", "F", "G", "I"])

    def test_solution_combination_with_identical_references(self):
        """Test identical reference curves in combined solution throws error"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        # Identical response reference.
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response_reference(f, np.ones_like(f), label="A")
        sol_a.add_noise_reference(f, np.ones_like(f), label="B")
        sol_b = Solution(f)
        sol_b.add_response(resp2)
        sol_b.add_response_reference(f, np.ones_like(f), label="A")
        self.assertRaises(ValueError, lambda: sol_a + sol_b)
        # Identical noise reference.
        sol_c = Solution(f)
        sol_c.add_response(resp1)
        sol_c.add_response_reference(f, np.ones_like(f), label="A")
        sol_c.add_noise_reference(f, np.ones_like(f), label="B")
        sol_d = Solution(f)
        sol_d.add_response(resp2)
        sol_d.add_noise_reference(f, np.ones_like(f), label="B")
        self.assertRaises(ValueError, lambda: sol_c + sol_d)
