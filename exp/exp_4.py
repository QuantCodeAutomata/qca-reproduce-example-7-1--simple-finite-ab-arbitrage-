"""
Experiment 4: Verify Propositions 2, 3, and 4.

Proposition 2: Trivial distortions (dF ≡ 1 for all arrows) ⟹ trivial holonomy (Hol ≡ 1).
Proposition 3: AB trading strategy V^AB(ω) = max(1, Hol(γ)(ω)) ≥ 1 everywhere.
Proposition 4: Weak AB strategy V^wAB ≥ 1 when reverse execution is admissible.

All computations use exact rational arithmetic (fractions.Fraction).
"""

from fractions import Fraction
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.holonomy import (
    ProbSpace,
    distortion,
    simplicial_distortion,
    compute_ab_strategy,
    compute_weak_ab_strategy,
    check_normalization,
    compute_ab_metrics,
    pointwise_product,
)
from exp.exp_1 import build_example_71
from exp.exp_2 import build_example_72


# ---------------------------------------------------------------------------
# Phase 1: Proposition 2 — Measure-Preserving Baseline
# ---------------------------------------------------------------------------

def build_measure_preserving_baseline() -> tuple[dict, dict]:
    """
    Construct a filtration where every F(i_k) is measure-preserving.
    Expected: dF(i_k) ≡ 1 for all k, Hol(γ) ≡ 1.

    Ω_{t0}={0,1}, μ_0={0:1/2, 1:1/2}
    Ω_{t1}={a,b}, μ_1={a:1/2, b:1/2}
    Ω_{t2}={x,y}, μ_2={x:1/2, y:1/2}

    F(i1): a↦0, b↦1   (measure-preserving)
    F(i2): x↦a, y↦b   (measure-preserving)
    F(i3): 0↦x, 1↦y   (measure-preserving)
    """
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
    return spaces, arrow_info


def verify_proposition_2() -> dict:
    """Verify Proposition 2: measure-preserving filtration ⟹ Hol ≡ 1."""
    print("\n--- Proposition 2: Trivial Distortion → Trivial Holonomy ---")
    spaces, arrow_info = build_measure_preserving_baseline()
    mu_0 = spaces["t0"].mu
    mu_1 = spaces["t1"].mu
    mu_2 = spaces["t2"].mu

    # Verify measure-preservation for each arrow
    for arrow_name, info in arrow_info.items():
        F_map = info["F_map"]
        source_sp = spaces[info["source_space"]]
        target_sp = spaces[info["target_space"]]
        pushforward = {y: Fraction(0) for y in target_sp.omega}
        for x in source_sp.omega:
            pushforward[F_map[x]] += source_sp.mu[x]
        assert pushforward == dict(target_sp.mu), (
            f"Arrow {arrow_name} is NOT measure-preserving: "
            f"pushforward={pushforward}, target_mu={dict(target_sp.mu)}"
        )
        print(f"  Arrow {arrow_name}: measure-preserving ✓")

    # Compute distortions
    dF_i1 = distortion("i1", spaces, arrow_info)
    dF_i2 = distortion("i2", spaces, arrow_info)
    dF_i3 = distortion("i3", spaces, arrow_info)
    print(f"  dF(i1) = {dF_i1}")
    print(f"  dF(i2) = {dF_i2}")
    print(f"  dF(i3) = {dF_i3}")

    assert all(v == Fraction(1) for v in dF_i1.values()), f"dF(i1) not ≡ 1: {dF_i1}"
    assert all(v == Fraction(1) for v in dF_i2.values()), f"dF(i2) not ≡ 1: {dF_i2}"
    assert all(v == Fraction(1) for v in dF_i3.values()), f"dF(i3) not ≡ 1: {dF_i3}"
    print("  All dF(i_k) ≡ 1 ✓")

    # Compute holonomy
    Hol_mp = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )
    print(f"  Hol(γ) = {Hol_mp}")
    assert all(v == Fraction(1) for v in Hol_mp.values()), (
        f"Hol not ≡ 1 for measure-preserving filtration: {Hol_mp}"
    )
    print("  Hol(γ) ≡ 1 ✓")

    metrics = compute_ab_metrics(Hol_mp, mu_0)
    assert metrics["P_AB"] == Fraction(0), f"P(Hol>1) = {metrics['P_AB']} (expected 0)"
    assert metrics["P_weak_AB"] == Fraction(0), (
        f"P(Hol≠1) = {metrics['P_weak_AB']} (expected 0)"
    )
    print(f"  P(Hol > 1) = {metrics['P_AB']} ✓  (no AB arbitrage)")
    print(f"  P(Hol ≠ 1) = {metrics['P_weak_AB']} ✓  (no weak AB arbitrage)")
    print("  Proposition 2 VERIFIED ✓")

    # Recursive trace explanation
    # tail_dist = dF_i2 * CE(i2, dF_i3) = 1 * CE(i2, ones) = 1 * 1 = 1
    # Hol = dF_i1 * CE(i1, ones) = 1 * 1 = 1
    print("\n  Recursive trace (why Hol ≡ 1):")
    print("    dF(i3) ≡ 1 → CE(i2, dF_i3) = CE(i2, 1) = dF(i2) = 1 → tail=1·1=1")
    print("    dF(i1) ≡ 1 → Hol = dF(i1) · CE(i1, tail) = 1 · 1 = 1 ✓")

    _, V_AB, E_V_AB, _ = compute_ab_strategy(Hol_mp, mu_0)
    assert E_V_AB == Fraction(1), f"E[V^AB] = {E_V_AB} (expected 1, no profit)"
    print(f"  E[V^AB] = {E_V_AB} ✓  (no profit from trivial holonomy)")

    return {
        "dF_i1": dF_i1, "dF_i2": dF_i2, "dF_i3": dF_i3,
        "Hol_mp": Hol_mp, "metrics": metrics,
        "spaces": spaces, "arrow_info": arrow_info,
    }


# ---------------------------------------------------------------------------
# Phase 2: Proposition 3 — Non-Negative Payoff
# ---------------------------------------------------------------------------

def verify_proposition_3_example_71() -> dict:
    """Verify Proposition 3 for Example 7.1: V^AB = max(1, Hol) ≥ 1."""
    print("\n--- Proposition 3 [Example 7.1]: V^AB ≥ 1 everywhere ---")
    spaces, arrow_info = build_example_71()
    mu_0 = spaces["t0"].mu
    Hol = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )
    theta_AB, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol, mu_0)
    print(f"  Hol(γ) = {Hol}")
    print(f"  θ^AB   = {theta_AB}")
    print(f"  V^AB   = {V_AB}")

    # Proposition 3: V^AB(ω) ≥ 1 for all ω
    assert all(V_AB[w] >= Fraction(1) for w in Hol), "V^AB < 1 at some state"
    print("  V^AB(ω) ≥ 1 everywhere ✓")

    # Verify V^AB = max(1, Hol)
    for w in Hol:
        expected = max(Fraction(1), Hol[w])
        assert V_AB[w] == expected, f"V^AB({w})={V_AB[w]} ≠ max(1,Hol)={expected}"
    print("  V^AB(ω) = max(1, Hol(γ)(ω)) ✓")

    # State-level checks
    assert V_AB[0] == Fraction(1), f"V^AB(0) should be 1 (Hol=0, θ=0)"
    assert V_AB[1] == Fraction(4), f"V^AB(1) should be 4 (Hol=4, θ=1)"
    print("  V^AB(0) = 1 ✓ (no execution when Hol=0)")
    print("  V^AB(1) = 4 ✓ (profit when Hol=4 > 1)")

    for w in Hol:
        if Hol[w] > Fraction(1):
            assert V_AB[w] > Fraction(1), f"V^AB({w}) not > 1 when Hol={Hol[w]}"
        else:
            assert V_AB[w] == Fraction(1), f"V^AB({w}) ≠ 1 when Hol={Hol[w]} ≤ 1"
    print("  V^AB > 1 ⟺ Hol > 1 ✓")
    print("  Proposition 3 VERIFIED for Example 7.1 ✓")
    return {"Hol": Hol, "V_AB": V_AB, "E_V_AB": E_V_AB}


def verify_proposition_3_example_72() -> dict:
    """Verify Proposition 3 for Example 7.2: V^AB = max(1, Hol) ≥ 1."""
    print("\n--- Proposition 3 [Example 7.2]: V^AB ≥ 1 everywhere ---")
    spaces, arrow_info = build_example_72()
    mu_0 = spaces["t0"].mu
    Hol = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )
    theta_AB, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol, mu_0)
    print(f"  Hol(γ) = {Hol}")
    print(f"  V^AB   = {V_AB}")

    assert all(V_AB[w] >= Fraction(1) for w in Hol), "V^AB < 1 at some state"
    print("  V^AB(ω) ≥ 1 everywhere ✓")

    for w in Hol:
        expected = max(Fraction(1), Hol[w])
        assert V_AB[w] == expected, f"V^AB({w})={V_AB[w]} ≠ max(1,Hol)={expected}"
    print("  V^AB(ω) = max(1, Hol(γ)(ω)) ✓")

    # Stronger condition: Hol ≥ 1 everywhere, so no-loss even without indicator
    assert all(Fraction(v) >= Fraction(1) for v in Hol.values()), (
        "Stronger AB: Hol < 1 at some state"
    )
    print("  min(Hol) = 1 ≥ 1 ✓ (stronger AB: no loss from direct loop execution)")
    print("  Proposition 3 VERIFIED for Example 7.2 ✓")
    return {"Hol": Hol, "V_AB": V_AB, "E_V_AB": E_V_AB}


# ---------------------------------------------------------------------------
# Phase 3: Proposition 4 — Weak AB Strategy
# ---------------------------------------------------------------------------

def verify_proposition_4_example_72() -> list[dict]:
    """Verify Proposition 4 for Example 7.2 (admissible case)."""
    print("\n--- Proposition 4 [Example 7.2]: Weak AB V^wAB ≥ 1 ---")
    spaces, arrow_info = build_example_72()
    mu_0 = spaces["t0"].mu
    Hol = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )

    rows = []
    for eps in [Fraction(0), Fraction(1, 10), Fraction(1, 2)]:
        theta_wAB, V_wAB = compute_weak_ab_strategy(Hol, mu_0, eps)
        E_V_wAB = sum(V_wAB[w] * mu_0[w] for w in Hol)
        print(f"  ε={eps}: θ^wAB={theta_wAB}, V^wAB={V_wAB}, E[V^wAB]={E_V_wAB}")
        assert all(V_wAB[w] >= Fraction(1) for w in Hol), (
            f"V^wAB < 1 for ε={eps}"
        )
        # At ω=0: Hol[0]=1 → not < 1-ε (for ε≥0) → θ^wAB(0)=0
        assert theta_wAB[0] == Fraction(0), (
            f"θ^wAB(0) should be 0 (Hol=1, no deviation) for ε={eps}"
        )
        rows.append({
            "ε": str(eps), "θ^wAB": str(theta_wAB), "V^wAB": str(V_wAB),
            "E[V^wAB]": str(E_V_wAB), "min(V^wAB) ≥ 1": "YES",
        })
    print("  Proposition 4 VERIFIED for Example 7.2 ✓ (all ε, V^wAB ≥ 1)")
    return rows


def document_admissibility_failure_71() -> None:
    """Document admissibility failure of weak AB at ω=0 in Example 7.1."""
    print("\n--- Proposition 4 [Example 7.1]: Admissibility Issue at ω=0 ---")
    spaces, arrow_info = build_example_71()
    mu_0 = spaces["t0"].mu
    Hol = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )

    eps = Fraction(0)
    theta_wAB, V_wAB = compute_weak_ab_strategy(Hol, mu_0, eps)
    print(f"  Hol(γ) = {Hol}")
    print(f"  ε=0: θ^wAB = {theta_wAB}")
    print(f"  θ^wAB(0) = {theta_wAB[0]}  ← INADMISSIBLE")
    print()
    print("  ADMISSIBILITY VIOLATION at ω=0:")
    print(f"    Hol(γ)(0) = {Hol[0]}")
    print(f"    θ^wAB(0) = -1  (reverse execution requested)")
    print(f"    Reverse loop holonomy = 1/Hol(γ)(0) = 1/0 → UNDEFINED")
    print("    → Reverse execution is INADMISSIBLE at ω=0")
    print("    → Proposition 4 does NOT apply at ω=0 (Remark 16 in paper)")
    print()
    print("  Naïve V^wAB(0) = 1 + (-1)*(0-1) = 2 (formally positive but inadmissible)")
    print(f"  V^wAB(0) [naive] = {V_wAB[0]}")
    print(f"  V^wAB(1) = {V_wAB[1]} ✓ (admissible, Hol=4>0)")
    print("  Proposition 4 PARTIALLY applicable in Example 7.1 (only at ω=1) ✓")


# ---------------------------------------------------------------------------
# Full Experiment 4 Runner
# ---------------------------------------------------------------------------

def run_experiment_4() -> dict:
    """Run all Proposition verifications."""
    print("=" * 60)
    print("EXPERIMENT 4: Propositions 2, 3, 4 Verification")
    print("=" * 60)

    r2 = verify_proposition_2()
    r3_71 = verify_proposition_3_example_71()
    r3_72 = verify_proposition_3_example_72()
    r4_rows = verify_proposition_4_example_72()
    document_admissibility_failure_71()

    print("\n" + "=" * 60)
    print("ALL PROPOSITION VERIFICATIONS PASSED ✓")
    print("=" * 60)

    return {
        "proposition_2": r2,
        "proposition_3_71": r3_71,
        "proposition_3_72": r3_72,
        "proposition_4_rows": r4_rows,
    }


def plot_propositions(results: dict, save_path: str) -> None:
    """Plot a 2×2 summary of Proposition verifications."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Propositions 2–4 Verification", fontsize=14, fontweight="bold")

    # Panel 1: Proposition 2 — trivial holonomy
    ax = axes[0, 0]
    mp_Hol = results["proposition_2"]["Hol_mp"]
    states = list(mp_Hol.keys())
    ax.bar([str(s) for s in states], [float(mp_Hol[s]) for s in states],
           color="lightgreen", edgecolor="black", width=0.4)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Prop 2: Measure-Preserving → Hol ≡ 1", fontsize=10)
    ax.set_xlabel("State ω"); ax.set_ylabel("Hol(γ)(ω)")
    ax.set_ylim(0, 1.5)

    # Panel 2: Proposition 3 — Example 7.1 payoff
    ax = axes[0, 1]
    V71 = results["proposition_3_71"]["V_AB"]
    Hol71 = results["proposition_3_71"]["Hol"]
    st71 = list(V71.keys())
    ax.bar([str(s) for s in st71], [float(V71[s]) for s in st71],
           color="coral", edgecolor="black", width=0.4, label="V^AB")
    ax.bar([str(s) for s in st71], [float(Hol71[s]) for s in st71],
           color="steelblue", edgecolor="black", width=0.4,
           alpha=0.5, label="Hol(γ)")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="V=1")
    ax.set_title("Prop 3 [Ex 7.1]: V^AB = max(1, Hol) ≥ 1", fontsize=10)
    ax.set_xlabel("State ω"); ax.set_ylabel("Value")
    ax.legend(fontsize=8)

    # Panel 3: Proposition 3 — Example 7.2 payoff
    ax = axes[1, 0]
    V72 = results["proposition_3_72"]["V_AB"]
    Hol72 = results["proposition_3_72"]["Hol"]
    st72 = list(V72.keys())
    ax.bar([str(s) for s in st72], [float(V72[s]) for s in st72],
           color="coral", edgecolor="black", width=0.4, label="V^AB")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="V=1")
    ax.set_title("Prop 3 [Ex 7.2]: V^AB ≥ 1 (Stronger AB)", fontsize=10)
    ax.set_xlabel("State ω"); ax.set_ylabel("V^AB(ω)")
    ax.legend(fontsize=8)

    # Panel 4: Proposition 4 — Weak AB strategy payoffs
    ax = axes[1, 1]
    rows = results["proposition_4_rows"]
    epsilons_labels = [r["ε"] for r in rows]
    e_vals = [float(eval(r["E[V^wAB]"].replace(" ", ""))) for r in rows]
    ax.bar(epsilons_labels, e_vals, color="plum", edgecolor="black", width=0.3)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="E=1")
    ax.set_title("Prop 4 [Ex 7.2]: E[V^wAB] by ε", fontsize=10)
    ax.set_xlabel("ε"); ax.set_ylabel("E[V^wAB]")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


if __name__ == "__main__":
    results = run_experiment_4()
    os.makedirs("results", exist_ok=True)
    plot_propositions(results, "results/exp4_propositions.png")
    print("\nExperiment 4 complete.")
