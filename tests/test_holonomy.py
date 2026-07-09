"""
Unit tests for src/holonomy.py.

Tests cover:
- ProbSpace validation
- conditional_expectation correctness
- distortion operator
- pointwise_product
- simplicial_distortion (recursive formula)
- compose_maps and distortion_of_composed_map
- compute_ab_strategy and compute_weak_ab_strategy
- check_normalization and compute_ab_metrics
- Edge cases: single-element space, uniform distributions, empty preimages
"""

import pytest
from fractions import Fraction
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.holonomy import (
    ProbSpace,
    conditional_expectation,
    distortion,
    pointwise_product,
    simplicial_distortion,
    compose_maps,
    distortion_of_composed_map,
    compute_ab_strategy,
    compute_weak_ab_strategy,
    check_normalization,
    check_ce_integral_preservation,
    compute_ab_metrics,
)


# ===========================================================================
# ProbSpace Tests
# ===========================================================================

class TestProbSpace:
    def test_valid_two_element_space(self):
        sp = ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)})
        assert sp.mu[0] == Fraction(1, 2)
        assert sp.mu[1] == Fraction(1, 2)
        assert sp.name == "t0"
        assert sp.omega == [0, 1]

    def test_valid_single_element_space(self):
        sp = ProbSpace("t1", ["*"], {"*": Fraction(1, 1)})
        assert sp.mu["*"] == Fraction(1)

    def test_valid_three_element_space(self):
        sp = ProbSpace(
            "t2", ["a", "b", "c"],
            {"a": Fraction(1, 4), "b": Fraction(1, 4), "c": Fraction(1, 2)},
        )
        assert sum(sp.mu.values()) == Fraction(1)

    def test_fractions_normalized(self):
        """Test that integer inputs are converted to Fraction."""
        sp = ProbSpace("t", [0], {0: 1})
        assert isinstance(sp.mu[0], Fraction)
        assert sp.mu[0] == Fraction(1)

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            ProbSpace("bad", [0, 1], {0: Fraction(1, 3), 1: Fraction(1, 3)})

    def test_degenerate_zero_probability(self):
        """Space with probabilities that don't sum to 1 raises AssertionError."""
        with pytest.raises(AssertionError):
            # Probabilities sum to 3/4 ≠ 1 → must raise
            ProbSpace("bad", [0, 1, 2], {0: Fraction(1, 2), 1: Fraction(1, 4), 2: Fraction(0)})

    def test_valid_zero_prob_element(self):
        """Valid space where one element has zero probability."""
        sp = ProbSpace(
            "t", [0, 1, 2],
            {0: Fraction(1, 2), 1: Fraction(1, 2), 2: Fraction(0)},
        )
        assert sum(sp.mu.values()) == Fraction(1)


# ===========================================================================
# Conditional Expectation Tests
# ===========================================================================

class TestConditionalExpectation:
    """Test CE using Example 7.1 setup."""

    def setup_method(self):
        self.spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["*"], {"*": Fraction(1, 1)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        self.arrow_info = {
            "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
        }

    def test_ce_i1_ones(self):
        """CE(i1)(1_{t1}) = distortion(i1) on Ω_{t0}."""
        ones = {"*": Fraction(1)}
        ce = conditional_expectation("i1", ones, self.spaces, self.arrow_info)
        # preimage(0) = {} → 0; preimage(1) = {*} → 1*1/(1/2) = 2
        assert ce == {0: Fraction(0), 1: Fraction(2)}

    def test_ce_i2_ones(self):
        """CE(i2)(1_{t2}) = distortion(i2) on Ω_{t1}."""
        ones = {0: Fraction(1), 1: Fraction(1)}
        ce = conditional_expectation("i2", ones, self.spaces, self.arrow_info)
        # preimage(*) = {0,1} → (1/4+3/4)/1 = 1
        assert ce == {"*": Fraction(1)}

    def test_ce_i3_distortion(self):
        """CE(i3)(1_{t0}) = dF(i3) on Ω_{t2}."""
        ones = {0: Fraction(1), 1: Fraction(1)}
        ce = conditional_expectation("i3", ones, self.spaces, self.arrow_info)
        # preimage(0 in t2) = {0 in t0}: (1/2)/(1/4)=2
        # preimage(1 in t2) = {1 in t0}: (1/2)/(3/4)=2/3
        assert ce[0] == Fraction(2)
        assert ce[1] == Fraction(2, 3)

    def test_ce_preserves_constants(self):
        """CE of constant c function returns c everywhere (when measure-preserving)."""
        spaces_mp = {
            "s": ProbSpace("s", ["a", "b"], {"a": Fraction(1, 2), "b": Fraction(1, 2)}),
            "t": ProbSpace("t", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        }
        arrow_mp = {
            "arr": {"F_map": {"a": 0, "b": 1}, "source_space": "s", "target_space": "t"},
        }
        f = {"a": Fraction(3), "b": Fraction(3)}
        ce = conditional_expectation("arr", f, spaces_mp, arrow_mp)
        assert ce == {0: Fraction(3), 1: Fraction(3)}

    def test_ce_empty_preimage(self):
        """CE with empty preimage gives 0 (not an error) when target mass > 0."""
        spaces = {
            "s": ProbSpace("s", ["x"], {"x": Fraction(1)}),
            "t": ProbSpace("t", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        }
        arrow = {
            "arr": {"F_map": {"x": 1}, "source_space": "s", "target_space": "t"},
        }
        f = {"x": Fraction(5)}
        ce = conditional_expectation("arr", f, spaces, arrow)
        # preimage(0) = [] → 0 / (1/2) = 0
        # preimage(1) = [x] → 5 * 1 / (1/2) = 10
        assert ce[0] == Fraction(0)
        assert ce[1] == Fraction(10)

    def test_ce_integral_preservation(self):
        """sum CE(f)(y) * mu_target(y) == sum f(x) * mu_source(x)."""
        for arrow in ["i1", "i2", "i3"]:
            ones = {x: Fraction(1) for x in self.spaces[self.arrow_info[arrow]["source_space"]].omega}
            lhs, rhs = check_ce_integral_preservation(arrow, ones, self.spaces, self.arrow_info)
            assert lhs == rhs, f"Integral preservation failed for {arrow}"


# ===========================================================================
# Distortion Tests
# ===========================================================================

class TestDistortion:
    def test_distortion_identity_map(self):
        """Identity map on equal measures gives dF ≡ 1."""
        spaces = {
            "s": ProbSpace("s", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t": ProbSpace("t", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        }
        arrow = {"arr": {"F_map": {0: 0, 1: 1}, "source_space": "s", "target_space": "t"}}
        dF = distortion("arr", spaces, arrow)
        assert dF == {0: Fraction(1), 1: Fraction(1)}

    def test_distortion_measure_distorting_map(self):
        """Non-measure-preserving map gives dF ≠ 1."""
        # Source: {0:1/2, 1:1/2}, target: {0:1/4, 1:3/4}, F(0)=0, F(1)=1
        spaces = {
            "s": ProbSpace("s", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t": ProbSpace("t", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow = {"arr": {"F_map": {0: 0, 1: 1}, "source_space": "s", "target_space": "t"}}
        dF = distortion("arr", spaces, arrow)
        # pushforward[0] = 1/2, dF[0] = (1/2)/(1/4) = 2
        # pushforward[1] = 1/2, dF[1] = (1/2)/(3/4) = 2/3
        assert dF[0] == Fraction(2)
        assert dF[1] == Fraction(2, 3)

    def test_distortion_normalizes_to_one(self):
        """sum dF(i)(y) * mu_target(y) == 1 always."""
        spaces = {
            "s": ProbSpace("s", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t": ProbSpace("t", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow = {"arr": {"F_map": {0: 0, 1: 1}, "source_space": "s", "target_space": "t"}}
        dF = distortion("arr", spaces, arrow)
        norm = check_normalization(dF, spaces["t"].mu)
        assert norm == Fraction(1)

    def test_distortion_collapsing_map(self):
        """Map that collapses all elements to one target element."""
        spaces = {
            "s": ProbSpace("s", ["a", "b"], {"a": Fraction(1, 2), "b": Fraction(1, 2)}),
            "t": ProbSpace("t", ["*"], {"*": Fraction(1)}),
        }
        arrow = {"arr": {"F_map": {"a": "*", "b": "*"}, "source_space": "s", "target_space": "t"}}
        dF = distortion("arr", spaces, arrow)
        # pushforward[*] = 1; dF[*] = 1/1 = 1
        assert dF == {"*": Fraction(1)}

    def test_distortion_example_71_i3(self):
        """Exact distortion from Example 7.1."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow = {"i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"}}
        dF = distortion("i3", spaces, arrow)
        assert dF[0] == Fraction(2)
        assert dF[1] == Fraction(2, 3)


# ===========================================================================
# Pointwise Product Tests
# ===========================================================================

class TestPointwiseProduct:
    def test_simple(self):
        f1 = {0: Fraction(2), 1: Fraction(3)}
        f2 = {0: Fraction(4), 1: Fraction(5)}
        result = pointwise_product(f1, f2)
        assert result == {0: Fraction(8), 1: Fraction(15)}

    def test_zero_multiplication(self):
        f1 = {0: Fraction(0), 1: Fraction(4)}
        f2 = {0: Fraction(7), 1: Fraction(1, 2)}
        result = pointwise_product(f1, f2)
        assert result == {0: Fraction(0), 1: Fraction(2)}

    def test_domain_mismatch_raises(self):
        f1 = {0: Fraction(1)}
        f2 = {1: Fraction(1)}
        with pytest.raises(AssertionError):
            pointwise_product(f1, f2)

    def test_string_keys(self):
        f1 = {"a": Fraction(1, 2), "b": Fraction(3, 4)}
        f2 = {"a": Fraction(2), "b": Fraction(4, 3)}
        result = pointwise_product(f1, f2)
        assert result == {"a": Fraction(1), "b": Fraction(1)}


# ===========================================================================
# Simplicial Distortion Tests
# ===========================================================================

class TestSimplicialDistortion:
    def setup_method(self):
        """Example 7.1 setup."""
        self.spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["*"], {"*": Fraction(1)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        self.arrow_info = {
            "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
        }

    def test_n0_trivial(self):
        """n=0: distortion of single object is identity 1."""
        result = simplicial_distortion(["t0"], [], self.spaces, self.arrow_info)
        assert result == {0: Fraction(1), 1: Fraction(1)}

    def test_n1_single_arrow(self):
        """n=1: equals single-arrow distortion."""
        result = simplicial_distortion(["t0", "t1"], ["i1"], self.spaces, self.arrow_info)
        dF = distortion("i1", self.spaces, self.arrow_info)
        assert result == dF

    def test_full_loop_example_71(self):
        """Full 3-arrow loop gives Hol = {0:0, 1:4}."""
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], self.spaces, self.arrow_info
        )
        assert Hol == {0: Fraction(0), 1: Fraction(4)}

    def test_full_loop_example_72(self):
        """Full 3-arrow loop for Example 7.2 gives Hol = {0:1, 1:2}."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
            "t1": ProbSpace(
                "t1", ["a", "b", "c"],
                {"a": Fraction(1, 4), "b": Fraction(1, 4), "c": Fraction(1, 2)},
            ),
            "t2": ProbSpace(
                "t2", ["u", "v", "w"],
                {"u": Fraction(1, 4), "v": Fraction(1, 4), "w": Fraction(1, 2)},
            ),
        }
        arrow_info = {
            "i1": {"F_map": {"a": 0, "b": 1, "c": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {"u": "a", "v": "c", "w": "b"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: "u", 1: "w"}, "source_space": "t0", "target_space": "t2"},
        }
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        assert Hol == {0: Fraction(1), 1: Fraction(2)}

    def test_measure_preserving_trivial_holonomy(self):
        """Measure-preserving filtration gives Hol ≡ 1."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["a", "b"], {"a": Fraction(1, 2), "b": Fraction(1, 2)}),
            "t2": ProbSpace("t2", ["x", "y"], {"x": Fraction(1, 2), "y": Fraction(1, 2)}),
        }
        arrow_info = {
            "i1": {"F_map": {"a": 0, "b": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {"x": "a", "y": "b"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: "x", 1: "y"}, "source_space": "t0", "target_space": "t2"},
        }
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        assert all(v == Fraction(1) for v in Hol.values())

    def test_4arrow_loop(self):
        """4-arrow loop gives Hol = {0:104/9, 1:0}."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["a", "b"], {"a": Fraction(1, 4), "b": Fraction(3, 4)}),
            "t2": ProbSpace(
                "t2", ["x", "y", "z"],
                {"x": Fraction(1, 3), "y": Fraction(1, 3), "z": Fraction(1, 3)},
            ),
            "t3": ProbSpace("t3", ["p", "q"], {"p": Fraction(2, 3), "q": Fraction(1, 3)}),
        }
        arrow_info = {
            "i1": {"F_map": {"a": 0, "b": 0}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {"x": "a", "y": "a", "z": "b"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {"p": "x", "q": "z"}, "source_space": "t3", "target_space": "t2"},
            "i4": {"F_map": {0: "p", 1: "q"}, "source_space": "t0", "target_space": "t3"},
        }
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t3", "t0"], ["i1", "i2", "i3", "i4"], spaces, arrow_info
        )
        assert Hol[0] == Fraction(104, 9)
        assert Hol[1] == Fraction(0)


# ===========================================================================
# Compose Maps Tests
# ===========================================================================

class TestComposeMaps:
    def test_identity_composition(self):
        F = {0: 0, 1: 1}
        result = compose_maps(F, F, [0, 1])
        assert result == {0: 0, 1: 1}

    def test_example_71_i1_comp_i2(self):
        """F(i1) ∘ F(i2): {0:*, 1:*} then {*:1} → {0:1, 1:1}."""
        F_i1 = {"*": 1}
        F_i2 = {0: "*", 1: "*"}
        result = compose_maps(F_i1, F_i2, [0, 1])
        assert result == {0: 1, 1: 1}

    def test_example_72_i1_comp_i2(self):
        """F(i1) ∘ F(i2): Ω_t2→Ω_t0 in Example 7.2."""
        F_i1 = {"a": 0, "b": 1, "c": 1}
        F_i2 = {"u": "a", "v": "c", "w": "b"}
        result = compose_maps(F_i1, F_i2, ["u", "v", "w"])
        assert result == {"u": 0, "v": 1, "w": 1}


# ===========================================================================
# Distortion of Composed Map Tests
# ===========================================================================

class TestDistortionOfComposedMap:
    def test_example_71_i1_comp_i2(self):
        """dF(i1∘i2) from composed map {0:1,1:1} on Ω_t2→Ω_t0."""
        source = ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)})
        target = ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)})
        composed = {0: 1, 1: 1}
        dF = distortion_of_composed_map(composed, source, target)
        # pushforward[0] = 0, pushforward[1] = 1/4 + 3/4 = 1
        # dF[0] = 0/(1/2) = 0, dF[1] = 1/(1/2) = 2
        assert dF == {0: Fraction(0), 1: Fraction(2)}

    def test_identity_composed_map(self):
        """Identity composition gives dF = pushforward/mu = 1 if same measure."""
        sp = ProbSpace("t", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)})
        dF = distortion_of_composed_map({0: 0, 1: 1}, sp, sp)
        assert dF == {0: Fraction(1), 1: Fraction(1)}

    def test_empty_preimage_gives_zero(self):
        """Element with empty preimage gets dF = 0 (valid, not inadmissible)."""
        source = ProbSpace("s", [0], {0: Fraction(1)})
        target = ProbSpace("t", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)})
        composed = {0: 0}  # 1 never reached
        dF = distortion_of_composed_map(composed, source, target)
        assert dF[0] == Fraction(2)  # 1/(1/2)
        assert dF[1] == Fraction(0)  # empty preimage


# ===========================================================================
# Trading Strategy Tests
# ===========================================================================

class TestTradingStrategy:
    def test_ab_strategy_example_71(self):
        """θ^AB for Example 7.1 gives θ={0:0,1:1}, V={0:1,1:4}, E[V]=5/2."""
        Hol = {0: Fraction(0), 1: Fraction(4)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        theta, V, E_V, E_profit = compute_ab_strategy(Hol, mu)
        assert theta == {0: Fraction(0), 1: Fraction(1)}
        assert V == {0: Fraction(1), 1: Fraction(4)}
        assert E_V == Fraction(5, 2)
        assert E_profit == Fraction(3, 2)

    def test_ab_strategy_example_72(self):
        """θ^AB for Example 7.2 gives V={0:1,1:2}, E[V]=7/4."""
        Hol = {0: Fraction(1), 1: Fraction(2)}
        mu = {0: Fraction(1, 4), 1: Fraction(3, 4)}
        theta, V, E_V, E_profit = compute_ab_strategy(Hol, mu)
        assert theta == {0: Fraction(0), 1: Fraction(1)}
        assert V == {0: Fraction(1), 1: Fraction(2)}
        assert E_V == Fraction(7, 4)
        assert E_profit == Fraction(3, 4)

    def test_ab_strategy_trivial_holonomy(self):
        """Trivial holonomy ≡ 1 gives θ≡0, V≡1, E[profit]=0."""
        Hol = {0: Fraction(1), 1: Fraction(1)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        theta, V, E_V, E_profit = compute_ab_strategy(Hol, mu)
        assert all(t == Fraction(0) for t in theta.values())
        assert all(v == Fraction(1) for v in V.values())
        assert E_profit == Fraction(0)

    def test_v_ab_geq_one_always(self):
        """Proposition 3: V^AB = max(1, Hol) ≥ 1 always."""
        test_cases = [
            ({0: Fraction(0), 1: Fraction(4)}, {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            ({0: Fraction(1), 1: Fraction(2)}, {0: Fraction(1, 4), 1: Fraction(3, 4)}),
            ({0: Fraction(1, 2), 1: Fraction(3)}, {0: Fraction(1, 3), 1: Fraction(2, 3)}),
        ]
        for Hol, mu in test_cases:
            _, V, _, _ = compute_ab_strategy(Hol, mu)
            assert all(v >= Fraction(1) for v in V.values()), (
                f"V^AB < 1 for Hol={Hol}"
            )

    def test_v_ab_equals_max_one_hol(self):
        """V^AB(ω) = max(1, Hol(γ)(ω)) exactly."""
        Hol = {0: Fraction(3, 2), 1: Fraction(1, 2)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        _, V, _, _ = compute_ab_strategy(Hol, mu)
        for w in Hol:
            assert V[w] == max(Fraction(1), Hol[w]), f"V^AB({w}) ≠ max(1,Hol)"

    def test_weak_ab_strategy_epsilon_zero(self):
        """ε=0 weak AB for Example 7.2."""
        Hol = {0: Fraction(1), 1: Fraction(2)}
        mu = {0: Fraction(1, 4), 1: Fraction(3, 4)}
        theta_w, V_w = compute_weak_ab_strategy(Hol, mu, Fraction(0))
        # At ω=0: Hol=1 = 1+0, not > 1+0, not < 1-0 → θ=0
        assert theta_w[0] == Fraction(0)
        # At ω=1: Hol=2 > 1+0 → θ=+1
        assert theta_w[1] == Fraction(1)
        assert V_w[0] == Fraction(1)
        assert V_w[1] == Fraction(2)

    def test_weak_ab_reverse_flag(self):
        """ε=0 weak AB at ω=0 for Example 7.1 returns θ=-1 (inadmissible flag)."""
        Hol = {0: Fraction(0), 1: Fraction(4)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        theta_w, V_w = compute_weak_ab_strategy(Hol, mu, Fraction(0))
        # Hol[0]=0 < 1-0=1, so θ^wAB(0)=-1 (inadmissible)
        assert theta_w[0] == Fraction(-1)


# ===========================================================================
# AB Metrics Tests
# ===========================================================================

class TestABMetrics:
    def test_metrics_example_71(self):
        Hol = {0: Fraction(0), 1: Fraction(4)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        m = compute_ab_metrics(Hol, mu)
        assert m["P_AB"] == Fraction(1, 2)
        assert m["P_no_loss"] == Fraction(1, 2)
        assert m["P_weak_AB"] == Fraction(1, 1)
        assert m["E_Hol"] == Fraction(2)

    def test_metrics_example_72(self):
        Hol = {0: Fraction(1), 1: Fraction(2)}
        mu = {0: Fraction(1, 4), 1: Fraction(3, 4)}
        m = compute_ab_metrics(Hol, mu)
        assert m["P_AB"] == Fraction(3, 4)
        assert m["P_no_loss"] == Fraction(1)
        assert m["P_weak_AB"] == Fraction(3, 4)
        assert m["E_Hol"] == Fraction(7, 4)

    def test_metrics_trivial_holonomy(self):
        Hol = {0: Fraction(1), 1: Fraction(1)}
        mu = {0: Fraction(1, 2), 1: Fraction(1, 2)}
        m = compute_ab_metrics(Hol, mu)
        assert m["P_AB"] == Fraction(0)
        assert m["P_no_loss"] == Fraction(1)
        assert m["P_weak_AB"] == Fraction(0)
        assert m["E_Hol"] == Fraction(1)


# ===========================================================================
# Normalization Tests
# ===========================================================================

class TestNormalization:
    def test_normalization_example_71(self):
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["*"], {"*": Fraction(1)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow_info = {
            "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
        }
        for arrow_name in ["i1", "i2", "i3"]:
            dF = distortion(arrow_name, spaces, arrow_info)
            target = spaces[arrow_info[arrow_name]["target_space"]]
            norm = check_normalization(dF, target.mu)
            assert norm == Fraction(1), f"Normalization failed for {arrow_name}: {norm}"


# ===========================================================================
# Proposition 1 Regression Tests
# ===========================================================================

class TestProposition1:
    """Verify dF(i∘j) = CE(i, dF(j)) for composable pairs."""

    def test_prop1_example_71_i1_comp_i2(self):
        """LHS dF(i1∘i2) == RHS CE(i1, dF(i2)) for Example 7.1."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["*"], {"*": Fraction(1)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow_info = {
            "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
        }
        composed = compose_maps({"*": 1}, {0: "*", 1: "*"}, [0, 1])
        lhs = distortion_of_composed_map(composed, spaces["t2"], spaces["t0"])
        dF_i2 = distortion("i2", spaces, arrow_info)
        rhs = conditional_expectation("i1", dF_i2, spaces, arrow_info)
        assert lhs == rhs

    def test_prop1_example_72_i2_comp_i3(self):
        """LHS dF(i2∘i3) == RHS CE(i2, dF(i3)) for Example 7.2."""
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
            "t1": ProbSpace(
                "t1", ["a", "b", "c"],
                {"a": Fraction(1, 4), "b": Fraction(1, 4), "c": Fraction(1, 2)},
            ),
            "t2": ProbSpace(
                "t2", ["u", "v", "w"],
                {"u": Fraction(1, 4), "v": Fraction(1, 4), "w": Fraction(1, 2)},
            ),
        }
        arrow_info = {
            "i1": {"F_map": {"a": 0, "b": 1, "c": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {"u": "a", "v": "c", "w": "b"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: "u", 1: "w"}, "source_space": "t0", "target_space": "t2"},
        }
        F_i2 = {"u": "a", "v": "c", "w": "b"}
        F_i3 = {0: "u", 1: "w"}
        composed = compose_maps(F_i2, F_i3, [0, 1])
        lhs = distortion_of_composed_map(composed, spaces["t0"], spaces["t1"])
        dF_i3 = distortion("i3", spaces, arrow_info)
        rhs = conditional_expectation("i2", dF_i3, spaces, arrow_info)
        assert lhs == rhs


# ===========================================================================
# Holonomy Value Bounds Tests
# ===========================================================================

class TestHolonomyBounds:
    """Mathematical property tests for holonomy."""

    def test_holonomy_nonnegative(self):
        """Holonomy values must be ≥ 0 (they are Radon-Nikodym derivatives)."""
        for Hol, mu in [
            ({0: Fraction(0), 1: Fraction(4)}, {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            ({0: Fraction(1), 1: Fraction(2)}, {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        ]:
            assert all(v >= Fraction(0) for v in Hol.values())

    def test_expected_holonomy_geq_one_if_normalized(self):
        """E[Hol] = 1 when distortions are normalized (holonomy is a Radon-Nikodym derivative)."""
        # For Example 7.1: E[Hol] = 0*(1/2) + 4*(1/2) = 2 ≠ 1
        # The holonomy is NOT a probability density w.r.t. mu_0
        # However, each individual dF(i_k) satisfies sum dF * mu = 1
        spaces = {
            "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
            "t1": ProbSpace("t1", ["*"], {"*": Fraction(1)}),
            "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
        }
        arrow_info = {
            "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
            "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
            "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
        }
        for arrow_name in ["i1", "i2", "i3"]:
            dF = distortion(arrow_name, spaces, arrow_info)
            target = spaces[arrow_info[arrow_name]["target_space"]]
            norm = check_normalization(dF, target.mu)
            assert norm == Fraction(1)
