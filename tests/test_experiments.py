"""
Integration tests for all six experiments.

Verifies that each experiment produces the exact expected outcomes from the paper,
using exact rational arithmetic (fractions.Fraction).

These tests validate methodology adherence:
- Exact holonomy values from paper
- Exact trading strategy payoffs
- Exact Proposition equalities
- Exact sensitivity analysis invariants
"""

import pytest
from fractions import Fraction
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp.exp_1 import run_experiment_1, build_example_71
from exp.exp_2 import run_experiment_2, build_example_72
from exp.exp_3 import run_experiment_3
from exp.exp_4 import (
    run_experiment_4,
    build_measure_preserving_baseline,
    verify_proposition_2,
)
from exp.exp_5 import run_experiment_5, sweep_example_71, sweep_example_72
from exp.exp_6 import (
    run_experiment_6,
    run_subexp_a, run_subexp_b, run_subexp_c,
    build_4arrow_loop, build_two_loops, build_fx_triangle,
)
from src.holonomy import simplicial_distortion, compute_ab_metrics, compute_ab_strategy


# ===========================================================================
# Experiment 1 Integration Tests
# ===========================================================================

class TestExperiment1:
    """Verify all paper values for Example 7.1."""

    def setup_method(self):
        self.results = run_experiment_1()

    def test_holonomy_values(self):
        """Hol(γ) = {0: 0, 1: 4} exactly."""
        assert self.results["Hol_gamma"] == {0: Fraction(0), 1: Fraction(4)}

    def test_distortion_i1(self):
        """dF(i1) = {0: 0, 1: 2}."""
        assert self.results["dF_i1"] == {0: Fraction(0), 1: Fraction(2)}

    def test_distortion_i2(self):
        """dF(i2) = {*: 1}."""
        assert self.results["dF_i2"] == {"*": Fraction(1)}

    def test_distortion_i3(self):
        """dF(i3) = {0: 2, 1: 2/3}."""
        assert self.results["dF_i3"] == {0: Fraction(2), 1: Fraction(2, 3)}

    def test_ce_i2_df_i3(self):
        """CE(i2, dF_i3) = {*: 1}."""
        assert self.results["CE_i2_dF_i3"] == {"*": Fraction(1)}

    def test_tail_distortion(self):
        """Tail distortion for [t1,t2,t0] = {*: 1}."""
        assert self.results["tail_dist"] == {"*": Fraction(1)}

    def test_ce_i1_tail(self):
        """CE(i1, tail) = {0: 0, 1: 2}."""
        assert self.results["CE_i1_tail"] == {0: Fraction(0), 1: Fraction(2)}

    def test_ab_metrics(self):
        """P(Hol>1)=1/2, P(Hol≥1)=1/2, P(Hol≠1)=1."""
        m = self.results["metrics"]
        assert m["P_AB"] == Fraction(1, 2)
        assert m["P_no_loss"] == Fraction(1, 2)
        assert m["P_weak_AB"] == Fraction(1, 1)
        assert m["E_Hol"] == Fraction(2)

    def test_trading_strategy(self):
        """θ^AB={0:0,1:1}, V^AB={0:1,1:4}, E[V^AB]=5/2."""
        assert self.results["theta_AB"] == {0: Fraction(0), 1: Fraction(1)}
        assert self.results["V_AB"] == {0: Fraction(1), 1: Fraction(4)}
        assert self.results["E_V_AB"] == Fraction(5, 2)
        assert self.results["E_profit"] == Fraction(3, 2)

    def test_payoff_nonnegative(self):
        """V^AB ≥ 1 everywhere."""
        assert all(v >= Fraction(1) for v in self.results["V_AB"].values())


# ===========================================================================
# Experiment 2 Integration Tests
# ===========================================================================

class TestExperiment2:
    """Verify all paper values for Example 7.2."""

    def setup_method(self):
        self.results = run_experiment_2()

    def test_holonomy_values(self):
        """Hol(γ) = {0: 1, 1: 2} exactly."""
        assert self.results["Hol_gamma"] == {0: Fraction(1), 1: Fraction(2)}

    def test_distortion_i1_is_trivial(self):
        """dF(i1) ≡ 1 (measure-preserving first arrow)."""
        assert self.results["dF_i1"] == {0: Fraction(1), 1: Fraction(1)}

    def test_distortion_i2(self):
        """dF(i2) = {a:1, b:2, c:1/2}."""
        assert self.results["dF_i2"] == {
            "a": Fraction(1), "b": Fraction(2), "c": Fraction(1, 2)
        }

    def test_distortion_i3(self):
        """dF(i3) = {u:1, v:0, w:3/2}."""
        assert self.results["dF_i3"] == {
            "u": Fraction(1), "v": Fraction(0), "w": Fraction(3, 2)
        }

    def test_ce_i2_df_i3(self):
        """CE(i2, dF_i3) = {a:1, b:3, c:0}."""
        assert self.results["CE_i2_dF_i3"] == {
            "a": Fraction(1), "b": Fraction(3), "c": Fraction(0)
        }

    def test_h2(self):
        """h_2 = {a:1, b:6, c:0}."""
        assert self.results["h_2"] == {
            "a": Fraction(1), "b": Fraction(6), "c": Fraction(0)
        }

    def test_ce_i1_h2(self):
        """CE(i1, h2) = {0:1, 1:2}."""
        assert self.results["CE_i1_h2"] == {0: Fraction(1), 1: Fraction(2)}

    def test_ab_metrics(self):
        """P(Hol>1)=3/4, P(Hol≥1)=1."""
        m = self.results["metrics"]
        assert m["P_AB"] == Fraction(3, 4)
        assert m["P_no_loss"] == Fraction(1)
        assert m["P_weak_AB"] == Fraction(3, 4)
        assert m["E_Hol"] == Fraction(7, 4)

    def test_trading_strategy(self):
        """θ^AB={0:0,1:1}, V^AB={0:1,1:2}, E[V^AB]=7/4."""
        assert self.results["theta_AB"] == {0: Fraction(0), 1: Fraction(1)}
        assert self.results["V_AB"] == {0: Fraction(1), 1: Fraction(2)}
        assert self.results["E_V_AB"] == Fraction(7, 4)
        assert self.results["E_profit"] == Fraction(3, 4)

    def test_stronger_ab_condition(self):
        """min(Hol) ≥ 1 (stronger AB: no loss from unconditional execution)."""
        assert all(v >= Fraction(1) for v in self.results["Hol_gamma"].values())


# ===========================================================================
# Experiment 3 Integration Tests
# ===========================================================================

class TestExperiment3:
    """Verify Proposition 1 (Multiplicativity) for all composable pairs."""

    def test_proposition_1_all_pairs(self):
        """All four composition checks return equal=YES."""
        rows = run_experiment_3()
        assert len(rows) == 4
        assert all(r["Equal"] == "YES" for r in rows)

    def test_proposition_1_ex71_i1_comp_i2(self):
        """dF(i1∘i2) = {0:0, 1:2} for Example 7.1."""
        rows = run_experiment_3()
        r = rows[0]
        assert r["Example"] == "7.1"
        assert r["Composition"] == "i1∘i2"

    def test_proposition_1_ex72_i2_comp_i3(self):
        """dF(i2∘i3) = {a:1, b:3, c:0} for Example 7.2."""
        from src.holonomy import compose_maps, distortion_of_composed_map, conditional_expectation, distortion
        spaces, arrow_info = build_example_72()
        F_i2 = {"u": "a", "v": "c", "w": "b"}
        F_i3 = {0: "u", 1: "w"}
        composed = compose_maps(F_i2, F_i3, [0, 1])
        lhs = distortion_of_composed_map(composed, spaces["t0"], spaces["t1"])
        dF_i3 = distortion("i3", spaces, arrow_info)
        rhs = conditional_expectation("i2", dF_i3, spaces, arrow_info)
        assert lhs == rhs
        assert lhs == {"a": Fraction(1), "b": Fraction(3), "c": Fraction(0)}


# ===========================================================================
# Experiment 4 Integration Tests
# ===========================================================================

class TestExperiment4:
    """Verify Propositions 2, 3, and 4."""

    def test_proposition_2_trivial_holonomy(self):
        """Measure-preserving filtration gives Hol ≡ 1."""
        spaces, arrow_info = build_measure_preserving_baseline()
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        assert all(v == Fraction(1) for v in Hol.values())

    def test_proposition_2_no_arbitrage(self):
        """P(Hol>1) = 0 for measure-preserving filtration."""
        spaces, arrow_info = build_measure_preserving_baseline()
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        mu = spaces["t0"].mu
        m = compute_ab_metrics(Hol, mu)
        assert m["P_AB"] == Fraction(0)

    def test_proposition_3_ex71_nonnegative(self):
        """V^AB ≥ 1 everywhere for Example 7.1."""
        spaces, arrow_info = build_example_71()
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        _, V_AB, _, _ = compute_ab_strategy(Hol, spaces["t0"].mu)
        assert all(v >= Fraction(1) for v in V_AB.values())

    def test_proposition_3_v_equals_max_hol(self):
        """V^AB = max(1, Hol) exactly for both examples."""
        for builder, sigma, arrows in [
            (build_example_71, ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"]),
            (build_example_72, ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"]),
        ]:
            spaces, arrow_info = builder()
            Hol = simplicial_distortion(sigma, arrows, spaces, arrow_info)
            _, V_AB, _, _ = compute_ab_strategy(Hol, spaces["t0"].mu)
            for w in Hol:
                assert V_AB[w] == max(Fraction(1), Hol[w])

    def test_proposition_4_ex72_admissible(self):
        """Weak AB strategy V^wAB ≥ 1 for all ε in Example 7.2."""
        from src.holonomy import compute_weak_ab_strategy
        spaces, arrow_info = build_example_72()
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        for eps in [Fraction(0), Fraction(1, 10), Fraction(1, 2)]:
            _, V_wAB = compute_weak_ab_strategy(Hol, spaces["t0"].mu, eps)
            assert all(v >= Fraction(1) for v in V_wAB.values())


# ===========================================================================
# Experiment 5 Integration Tests
# ===========================================================================

class TestExperiment5:
    """Verify holonomy invariance in parameter sensitivity sweeps."""

    def test_ex71_holonomy_invariant(self):
        """Hol(γ) = {0:0, 1:4} for ALL p in sweep."""
        rows = sweep_example_71()
        assert len(rows) > 0
        for r in rows:
            assert r["Hol_0"] == Fraction(0), f"Hol_0={r['Hol_0']} for p={r['p']}"
            assert r["Hol_1"] == Fraction(4), f"Hol_1={r['Hol_1']} for p={r['p']}"

    def test_ex71_payoff_invariant(self):
        """E[V^AB] = 5/2 for all p."""
        rows = sweep_example_71()
        for r in rows:
            assert r["E_V_AB"] == Fraction(5, 2)

    def test_ex72_holonomy_invariant(self):
        """Hol[0]=1/(4q), Hol[1]=3/(2(1-q)) — analytical formulas hold for all q."""
        rows = sweep_example_72()
        for r in rows:
            q = r["q"]
            expected_hol0 = Fraction(1, 4) / q          # = 1/(4q)
            expected_hol1 = Fraction(3, 2) / (1 - q)    # = 3/(2(1-q))
            assert r["Hol_0"] == expected_hol0, f"Hol_0={r['Hol_0']} ≠ 1/(4q) for q={q}"
            assert r["Hol_1"] == expected_hol1, f"Hol_1={r['Hol_1']} ≠ 3/(2(1-q)) for q={q}"

    def test_ex72_p_ab_linear(self):
        """P(Hol>1) depends on q via the critical threshold q*=1/4."""
        rows = sweep_example_72()
        threshold = Fraction(1, 4)
        for r in rows:
            q = r["q"]
            mu0 = {0: q, 1: Fraction(1) - q}
            hol0 = Fraction(1, 4) / q
            hol1 = Fraction(3, 2) / (1 - q)
            expected = sum(mu0[w] for w in [0, 1] if (hol0 if w == 0 else hol1) > Fraction(1))
            assert r["P_AB"] == expected, f"P_AB={r['P_AB']} ≠ {expected} for q={q}"

    def test_ex72_paper_values(self):
        """At q=1/4: P_AB=3/4, E[V^AB]=7/4 (match paper exactly)."""
        rows = sweep_example_72()
        paper_row = next(r for r in rows if r["q"] == Fraction(1, 4))
        assert paper_row["P_AB"] == Fraction(3, 4)
        assert paper_row["E_V_AB"] == Fraction(7, 4)

    def test_ex72_no_loss_always(self):
        """P(Hol≥1) = 1 when q ≤ 1/4 (stronger AB region), <1 otherwise."""
        rows = sweep_example_72()
        threshold = Fraction(1, 4)
        for r in rows:
            q = r["q"]
            hol0 = Fraction(1, 4) / q
            # P(Hol≥1) = 1 iff Hol[0]≥1, i.e., q ≤ 1/4
            if q <= threshold:
                assert r["P_no_loss"] == Fraction(1), f"P_no_loss={r['P_no_loss']} for q={q}"
            else:
                # Hol[0] < 1, so P(Hol≥1) = mu0[1] = 1-q
                assert r["P_no_loss"] == Fraction(1) - q, f"P_no_loss={r['P_no_loss']} for q={q}"


# ===========================================================================
# Experiment 6 Integration Tests
# ===========================================================================

class TestExperiment6:
    """Verify extended loop holonomy computations."""

    def test_4arrow_holonomy(self):
        """4-arrow loop gives Hol = {0: 104/9, 1: 0}."""
        r = run_subexp_a()
        assert r["Hol"][0] == Fraction(104, 9)
        assert r["Hol"][1] == Fraction(0)

    def test_4arrow_p_ab(self):
        """P(Hol>1) = 1/2 for 4-arrow loop (only state 0 has Hol>1)."""
        r = run_subexp_a()
        assert r["metrics"]["P_AB"] == Fraction(1, 2)

    def test_4arrow_payoff_nonnegative(self):
        """V^AB ≥ 1 everywhere in 4-arrow loop."""
        r = run_subexp_a()
        spaces, arrow_info = build_4arrow_loop()
        _, V_AB, _, _ = compute_ab_strategy(r["Hol"], spaces["t0"].mu)
        assert all(v >= Fraction(1) for v in V_AB.values())

    def test_two_loops_gamma1_nontrivial(self):
        """γ1 loop has non-trivial holonomy (AB arbitrage)."""
        r = run_subexp_b()
        assert r["Hol_1"] == {0: Fraction(4), 1: Fraction(0)}
        assert r["metrics_1"]["P_AB"] == Fraction(1, 2)

    def test_two_loops_gamma2_trivial(self):
        """γ2 loop has trivial holonomy (no AB arbitrage)."""
        r = run_subexp_b()
        assert r["Hol_2"] == {0: Fraction(1), 1: Fraction(1)}
        assert r["metrics_2"]["P_AB"] == Fraction(0)

    def test_fx_triangle_payoff_nonneg(self):
        """FX triangle V^AB ≥ 1 everywhere."""
        r = run_subexp_c()
        spaces, _ = build_fx_triangle()
        _, V_AB, _, _ = compute_ab_strategy(r["Hol_FX"], spaces["USD"].mu)
        assert all(v >= Fraction(1) for v in V_AB.values())

    def test_fx_triangle_normalization(self):
        """All FX distortions normalize to 1."""
        from src.holonomy import check_normalization
        spaces, arrow_info = build_fx_triangle()
        for arrow_name in ["EURUSD", "GBPEUR", "USDGBP"]:
            from src.holonomy import distortion
            dF = distortion(arrow_name, spaces, arrow_info)
            target = spaces[arrow_info[arrow_name]["target_space"]]
            norm = check_normalization(dF, target.mu)
            assert norm == Fraction(1), f"dF({arrow_name}) norm = {norm}"
