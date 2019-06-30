"""Data tests"""

import numpy as np
from zero.solution import Solution, matches_between
from zero.data import MultiNoiseDensity
from .data import ZeroDataTestCase


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
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
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

    def test_solution_combination_new_name(self):
        """Test new name in combined solution"""
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
        sol_d = sol_a + sol_b + sol_c
        self.assertEqual(sol_d.name, f"{sol_a.name} + {sol_b.name} + {sol_c.name}")

    def test_solution_combination_different_frequencies(self):
        """Test that combining solutions with different frequency vectors throws error."""
        f1 = self._freqs()
        f2 = self._freqs()
        sol_a = Solution(f1)
        sol_b = Solution(f2)
        self.assertRaises(ValueError, sol_a.combine, sol_b)

    def test_solution_combination_default_only(self):
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
        self.assertCountEqual(sol_d.functions[sol_a.name], [resp1])
        self.assertCountEqual(sol_d.functions[sol_b.name], [resp2])
        self.assertCountEqual(sol_d.functions[sol_c.name], [resp3])

    def test_solution_combination_mixed(self):
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
        self.assertCountEqual(sol_d.functions[sol_a.name], [resp1])
        self.assertCountEqual(sol_d.functions[f"a ({sol_a.name})"], [resp2])
        self.assertCountEqual(sol_d.functions[f"b ({sol_a.name})"], [resp3])
        self.assertCountEqual(sol_d.functions[sol_b.name], [resp4])
        self.assertCountEqual(sol_d.functions[f"a ({sol_b.name})"], [resp5])
        self.assertCountEqual(sol_d.functions[f"c ({sol_b.name})"], [resp6])
        self.assertCountEqual(sol_d.functions[sol_c.name], [resp7])
        self.assertCountEqual(sol_d.functions[f"a ({sol_c.name})"], [resp8])
        self.assertCountEqual(sol_d.functions[f"b ({sol_c.name})"], [resp9])
        self.assertCountEqual(sol_d.functions[f"c ({sol_c.name})"], [resp10])
        self.assertCountEqual(sol_d.functions[f"d ({sol_c.name})"], [resp11])

    def test_solution_combination_identical_responses_valid(self):
        """Test that combining solutions with identical responses in default group is valid."""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2, group="b")
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        sol_b.add_response(resp2, group="b")
        sol_c = sol_a.combine(sol_b)
        self.assertCountEqual(sol_c.functions[sol_a.name], [resp1])
        self.assertCountEqual(sol_c.functions[f"b ({sol_a.name})"], [resp2])
        self.assertCountEqual(sol_c.functions[sol_b.name], [resp1])
        self.assertCountEqual(sol_c.functions[f"b ({sol_b.name})"], [resp2])

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
        self.assertCountEqual(sol_c.functions[sol_a.name], [resp1, resp2])
        self.assertCountEqual(sol_c.functions[f"b ({sol_b.name})"], [resp1, resp2])

    def test_solution_combination_identical_noise_valid(self):
        """Test that combining solutions with identical noise in default group is valid."""
        f = self._freqs()
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2, group="b")
        sol_b = Solution(f)
        sol_b.add_noise(noise1)
        sol_b.add_noise(noise2, group="b")
        sol_c = sol_a.combine(sol_b)
        self.assertCountEqual(sol_c.functions[sol_a.name], [noise1])
        self.assertCountEqual(sol_c.functions[f"b ({sol_a.name})"], [noise2])
        self.assertCountEqual(sol_c.functions[sol_b.name], [noise1])
        self.assertCountEqual(sol_c.functions[f"b ({sol_b.name})"], [noise2])

    def test_solution_combination_identical_noise_different_group_valid(self):
        """Test that combining solutions with identical noise in different groups is valid."""
        f = self._freqs()
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2)
        sol_b = Solution(f)
        sol_b.add_noise(noise1, group="b")
        sol_b.add_noise(noise2, group="b")
        sol_c = sol_a.combine(sol_b)
        self.assertCountEqual(sol_c.functions[sol_a.name], [noise1, noise2])
        self.assertCountEqual(sol_c.functions[f"b ({sol_b.name})"], [noise1, noise2])

    def test_solution_combination_merge_groups_default_only(self):
        """Test default groups in combined solution when merge_groups is True."""
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
        sol_d = sol_a.combine(sol_b, sol_c, merge_groups=True)
        # Combined solution shouldn't have any new groups.
        self.assertCountEqual(sol_d.groups, sol_a.groups)
        self.assertCountEqual(sol_d.groups, sol_b.groups)
        self.assertCountEqual(sol_d.groups, sol_c.groups)
        # Check functions.
        self.assertCountEqual(sol_d.functions[sol_d.DEFAULT_GROUP_NAME], [resp1, resp2, resp3])

    def test_solution_combination_merge_groups_mixed(self):
        """Test mixed groups in combined solution when merge_groups is True."""
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
        sol_d = sol_a.combine(sol_b, sol_c, merge_groups=True)
        self.assertCountEqual(sol_d.groups, set(sol_a.groups + sol_b.groups + sol_c.groups))
        # Check functions.
        self.assertCountEqual(sol_d.functions[sol_d.DEFAULT_GROUP_NAME], [resp1, resp4, resp7])
        self.assertCountEqual(sol_d.functions["a"], [resp2, resp5, resp8])
        self.assertCountEqual(sol_d.functions["b"], [resp3, resp9])
        self.assertCountEqual(sol_d.functions["c"], [resp6, resp10])
        self.assertCountEqual(sol_d.functions["d"], [resp11])

    def test_solution_combination_merge_groups_identical_responses_invalid(self):
        """Test that combining solutions with identical responses in default group throws error
        when merge_groups is True"""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2, group="b")
        sol_b = Solution(f)
        sol_b.add_response(resp1)
        sol_b.add_response(resp2, group="b")
        self.assertRaises(ValueError, sol_a.combine, sol_b, merge_groups=True)

    def test_solution_combination_merge_groups_identical_responses_different_group_valid(self):
        """Test that combining solutions with identical responses in different groups is valid
        when merge_groups is True."""
        f = self._freqs()
        resp1 = self._i_i_response(f)
        resp2 = self._i_v_response(f)
        sol_a = Solution(f)
        sol_a.add_response(resp1)
        sol_a.add_response(resp2)
        sol_b = Solution(f)
        sol_b.add_response(resp1, group="b")
        sol_b.add_response(resp2, group="b")
        sol_c = sol_a.combine(sol_b, merge_groups=True)
        self.assertCountEqual(sol_c.functions[sol_c.DEFAULT_GROUP_NAME], [resp1, resp2])
        self.assertCountEqual(sol_c.functions["b"], [resp1, resp2])

    def test_solution_combination_merge_groups_identical_noise_invalid(self):
        """Test that combining solutions with identical noise in default group throws error
        when merge_groups is True."""
        f = self._freqs()
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2, group="b")
        sol_b = Solution(f)
        sol_b.add_noise(noise1)
        sol_b.add_noise(noise2, group="b")
        self.assertRaises(ValueError, sol_a.combine, sol_b, merge_groups=True)

    def test_solution_combination_merge_groups_identical_noise_different_group_valid(self):
        """Test that combining solutions with identical noise in different groups is valid
        when merge_groups is True."""
        f = self._freqs()
        noise1 = self._voltage_noise_at_node(f)
        noise2 = self._voltage_noise_at_node(f)
        sol_a = Solution(f)
        sol_a.add_noise(noise1)
        sol_a.add_noise(noise2)
        sol_b = Solution(f)
        sol_b.add_noise(noise1, group="b")
        sol_b.add_noise(noise2, group="b")
        sol_c = sol_a.combine(sol_b, merge_groups=True)
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

    def test_get_noise_with_degenerate_functions_same_source(self):
        f = self._freqs()
        res1 = self._resistor()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._voltage_noise_at_comp(f, source=noise1.source)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertRaises(ValueError, sol.get_noise, source=noise1.source)

    def test_get_noise_with_degenerate_functions_same_sink(self):
        f = self._freqs()
        res1 = self._resistor()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._voltage_noise_at_comp(f, sink=noise1.sink)
        sol = Solution(f)
        sol.add_noise(noise1)
        sol.add_noise(noise2)
        self.assertRaises(ValueError, sol.get_noise, sink=noise1.sink)

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

    def test_get_noise_sum_with_degenerate_functions_same_sink(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._voltage_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f, sink=noise1.sink)
        noise4 = self._voltage_noise_at_comp(f, sink=noise1.sink)
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
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f)
        sol = self._solution(f)
        sol.add_noise(noise1)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [noise1])
        sol.replace(noise1, noise2)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [noise2])

    def test_noise_replacement_with_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f)
        sol = self._solution(f)
        sol.add_noise(noise1, group="b")
        self.assertEqual(sol.functions["b"], [noise1])
        # Trying to replace without specifying group is not allowed.
        self.assertRaises(ValueError, sol.replace, noise1, noise2)
        sol.replace(noise1, noise2, group="b")
        self.assertEqual(sol.functions["b"], [noise2])

    def test_noise_sum_replacement_no_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f)
        noise4 = self._current_noise_at_comp(f, sink=noise3.sink)
        sum1 = self._multi_noise_density(noise1.sink, [noise1, noise2])
        sum2 = self._multi_noise_density(noise3.sink, [noise3, noise4])
        sol = self._solution(f)
        sol.add_noise_sum(sum1)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [sum1])
        sol.replace(sum1, sum2)
        self.assertEqual(sol.functions[sol.DEFAULT_GROUP_NAME], [sum2])

    def test_noise_sum_replacement_with_group(self):
        f = self._freqs()
        noise1 = self._voltage_noise_at_comp(f)
        noise2 = self._current_noise_at_comp(f, sink=noise1.sink)
        noise3 = self._voltage_noise_at_comp(f)
        noise4 = self._current_noise_at_comp(f, sink=noise3.sink)
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
        noise1 = self._voltage_noise_at_node(f, sink=node1)
        noise1.label = label1
        noise2 = self._voltage_noise_at_node(f, sink=node1)
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
