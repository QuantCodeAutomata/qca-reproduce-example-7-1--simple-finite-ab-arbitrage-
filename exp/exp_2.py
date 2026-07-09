"""
Experiment 2: Reproduce Example 7.2 — Stronger Finite AB Arbitrage with Hol(γ) = {0:1, 1:2}.

Three finite probability spaces with three-element sample spaces connected by measurable
maps forming a triangular loop. The holonomy is ≥ 1 everywhere (stronger AB condition).

Expected holonomy: Hol(γ) = {0: Fraction(1,1), 1: Fraction(2,1)}
Expected E[V^AB] = 7/4, E[profit] = 3/4
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
    conditional_expectation,
    distortion,
    pointwise_product,
    simplicial_distortion,
    compute_ab_strategy,
    compute_weak_ab_strategy,
    check_normalization,
    check_ce_integral_preservation,
    compute_ab_metrics,
)


# ---------------------------------------------------------------------------
# Define Input Structures
# ---------------------------------------------------------------------------

def build_example_72() -> tuple[dict, dict]:
    """
    Build probability spaces and arrow_info for Example 7.2.

    Ω_{t0} = {0,1},      μ_0 = {0:1/4, 1:3/4}
    Ω_{t1} = {a,b,c},   μ_1 = {a:1/4, b:1/4, c:1/2}
    Ω_{t2} = {u,v,w},   μ_2 = {u:1/4, v:1/4, w:1/2}

    F(i1): Ω_{t1}→Ω_{t0}: a↦0, b↦1, c↦1
    F(i2): Ω_{t2}→Ω_{t1}: u↦a, v↦c, w↦b
    F(i3): Ω_{t0}→Ω_{t2}: 0↦u, 1↦w
    """
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
    return spaces, arrow_info


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------

def run_experiment_2() -> dict:
    """Execute all phases of Experiment 2 and return results."""
    spaces, arrow_info = build_example_72()
    mu_0 = spaces["t0"].mu
    mu_1 = spaces["t1"].mu
    mu_2 = spaces["t2"].mu

    print("=" * 60)
    print("EXPERIMENT 2: Example 7.2 — Stronger AB Arbitrage, Hol = {0:1, 1:2}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step A: h3 = dF(i3) on Ω_{t2} = {u,v,w}
    # ----------------------------------------------------------------
    dF_i3 = distortion("i3", spaces, arrow_info)
    print("\nStep A: dF(i3) =", dF_i3)
    assert dF_i3 == {
        "u": Fraction(1, 1),
        "v": Fraction(0, 1),
        "w": Fraction(3, 2),
    }, f"dF_i3 mismatch: {dF_i3}"

    # ----------------------------------------------------------------
    # Step B: CE(i2, dF_i3) on Ω_{t1} = {a,b,c}
    # ----------------------------------------------------------------
    CE_i2_dF_i3 = conditional_expectation("i2", dF_i3, spaces, arrow_info)
    print("Step B: CE(i2, dF_i3) =", CE_i2_dF_i3)
    assert CE_i2_dF_i3 == {
        "a": Fraction(1, 1),
        "b": Fraction(3, 1),
        "c": Fraction(0, 1),
    }, f"CE_i2_dF_i3 mismatch: {CE_i2_dF_i3}"

    # ----------------------------------------------------------------
    # Step C: dF(i2) on Ω_{t1} = {a,b,c}
    # ----------------------------------------------------------------
    dF_i2 = distortion("i2", spaces, arrow_info)
    print("Step C: dF(i2) =", dF_i2)
    assert dF_i2 == {
        "a": Fraction(1, 1),
        "b": Fraction(2, 1),
        "c": Fraction(1, 2),
    }, f"dF_i2 mismatch: {dF_i2}"

    # ----------------------------------------------------------------
    # Step D: h2 = dF(i2) * CE(i2, dF_i3) on Ω_{t1}
    # ----------------------------------------------------------------
    h_2 = pointwise_product(dF_i2, CE_i2_dF_i3)
    print("Step D: h_2 =", h_2)
    assert h_2 == {
        "a": Fraction(1, 1),
        "b": Fraction(6, 1),
        "c": Fraction(0, 1),
    }, f"h_2 mismatch: {h_2}"

    # ----------------------------------------------------------------
    # Step E: CE(i1, h_2) on Ω_{t0} = {0,1}
    # ----------------------------------------------------------------
    CE_i1_h2 = conditional_expectation("i1", h_2, spaces, arrow_info)
    print("Step E: CE(i1, h_2) =", CE_i1_h2)
    assert CE_i1_h2 == {0: Fraction(1, 1), 1: Fraction(2, 1)}, (
        f"CE_i1_h2 mismatch: {CE_i1_h2}"
    )

    # ----------------------------------------------------------------
    # Step F: dF(i1) on Ω_{t0} — expect ≡ 1 (measure-preserving)
    # ----------------------------------------------------------------
    dF_i1 = distortion("i1", spaces, arrow_info)
    print("Step F: dF(i1) =", dF_i1)
    assert dF_i1 == {0: Fraction(1, 1), 1: Fraction(1, 1)}, (
        f"dF_i1 mismatch: {dF_i1}"
    )
    print("  → dF(i1) ≡ 1: i1 is measure-preserving ✓")

    # ----------------------------------------------------------------
    # Step G: Hol(γ) = dF(i1) * CE(i1, h_2)
    # ----------------------------------------------------------------
    Hol_manual = pointwise_product(dF_i1, CE_i1_h2)
    print("Step G: Hol(γ) [manual] =", Hol_manual)
    assert Hol_manual == {0: Fraction(1, 1), 1: Fraction(2, 1)}, (
        f"Holonomy mismatch: {Hol_manual}"
    )

    # ----------------------------------------------------------------
    # Phase 4: Recursive verification
    # ----------------------------------------------------------------
    sigma_loop = ["t0", "t1", "t2", "t0"]
    arrows_loop = ["i1", "i2", "i3"]
    Hol_gamma = simplicial_distortion(sigma_loop, arrows_loop, spaces, arrow_info)
    print("\nPhase 4: Hol(γ) [recursive] =", Hol_gamma)
    assert Hol_gamma == {0: Fraction(1, 1), 1: Fraction(2, 1)}, (
        f"Recursive holonomy mismatch: {Hol_gamma}"
    )

    # ----------------------------------------------------------------
    # Phase 5: AB Arbitrage Check
    # ----------------------------------------------------------------
    metrics = compute_ab_metrics(Hol_gamma, mu_0)
    print("\nPhase 5: AB Arbitrage Metrics")
    print(f"  P(Hol > 1) = {metrics['P_AB']}")
    print(f"  P(Hol >= 1) = {metrics['P_no_loss']}")
    print(f"  P(Hol != 1) = {metrics['P_weak_AB']}")
    print(f"  E[Hol] = {metrics['E_Hol']}")

    assert metrics["P_AB"] == Fraction(3, 4), f"P_AB = {metrics['P_AB']}"
    assert metrics["P_no_loss"] == Fraction(1, 1), f"P_no_loss = {metrics['P_no_loss']}"
    assert metrics["P_weak_AB"] == Fraction(3, 4), f"P_weak_AB = {metrics['P_weak_AB']}"
    assert metrics["E_Hol"] == Fraction(7, 4), f"E_Hol = {metrics['E_Hol']}"
    print("  Stronger AB condition: P(Hol ≥ 1) = 1 ✓ (no loss from direct execution)")

    # ----------------------------------------------------------------
    # Phase 6: Trading Strategy
    # ----------------------------------------------------------------
    theta_AB, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol_gamma, mu_0)
    print("\nPhase 6: Trading Strategy")
    print(f"  θ^AB = {theta_AB}")
    print(f"  V^AB = {V_AB}")
    print(f"  E[V^AB] = {E_V_AB}")
    print(f"  E[V^AB] - 1 = {E_profit}")

    assert theta_AB == {0: Fraction(0), 1: Fraction(1)}, f"theta_AB = {theta_AB}"
    assert V_AB == {0: Fraction(1, 1), 1: Fraction(2, 1)}, f"V_AB = {V_AB}"
    assert E_V_AB == Fraction(7, 4), f"E_V_AB = {E_V_AB}"
    assert E_profit == Fraction(3, 4), f"E_profit = {E_profit}"

    # ----------------------------------------------------------------
    # Phase 7: Weak AB Strategy (no inadmissibility issues)
    # ----------------------------------------------------------------
    print("\nPhase 7: Weak AB Strategy")
    for eps in [Fraction(0), Fraction(1, 10), Fraction(1, 2)]:
        theta_wAB, V_wAB = compute_weak_ab_strategy(Hol_gamma, mu_0, eps)
        print(f"  ε={eps}: θ^wAB={theta_wAB}, V^wAB={V_wAB}")
        # At ω=0: Hol=1, not > 1+ε or < 1-ε for ε≥0 → θ^wAB(0)=0
        assert theta_wAB[0] == Fraction(0), (
            f"At ω=0 (Hol=1), θ^wAB should be 0 for ε≥0"
        )
        # V^wAB >= 1 everywhere (no inadmissibility)
        assert all(V_wAB[w] >= Fraction(1) for w in Hol_gamma), (
            f"V^wAB < 1 at some state for ε={eps}"
        )
    print("  No inadmissibility issues (Hol ≥ 1 everywhere) ✓")

    # ----------------------------------------------------------------
    # Phase 8: Validation
    # ----------------------------------------------------------------
    print("\nPhase 8: Validation Checks")

    norm_i1 = check_normalization(dF_i1, mu_0)
    norm_i2 = check_normalization(dF_i2, mu_1)
    norm_i3 = check_normalization(dF_i3, mu_2)
    print(f"  Σ dF(i1)·μ_t0 = {norm_i1} {'✓' if norm_i1 == 1 else '✗'}")
    print(f"  Σ dF(i2)·μ_t1 = {norm_i2} {'✓' if norm_i2 == 1 else '✗'}")
    print(f"  Σ dF(i3)·μ_t2 = {norm_i3} {'✓' if norm_i3 == 1 else '✗'}")
    assert norm_i1 == Fraction(1)
    assert norm_i2 == Fraction(1)
    assert norm_i3 == Fraction(1)

    for arrow in ["i1", "i2", "i3"]:
        source_sp = spaces[arrow_info[arrow]["source_space"]]
        ones_f = {x: Fraction(1) for x in source_sp.omega}
        lhs, rhs = check_ce_integral_preservation(arrow, ones_f, spaces, arrow_info)
        print(f"  CE integral preservation {arrow}: {lhs} == {rhs} {'✓' if lhs == rhs else '✗'}")
        assert lhs == rhs

    assert all(V_AB[w] >= Fraction(1) for w in Hol_gamma), "V^AB < 1"
    print("  V^AB(ω) >= 1 everywhere ✓")
    assert min(Fraction(v) for v in Hol_gamma.values()) >= Fraction(1), (
        "min(Hol) < 1: stronger AB condition violated"
    )
    print("  min(Hol) >= 1 ✓ (stronger AB condition)")

    print("\nAll assertions PASSED for Experiment 2 ✓")

    return {
        "dF_i1": dF_i1,
        "dF_i2": dF_i2,
        "dF_i3": dF_i3,
        "CE_i2_dF_i3": CE_i2_dF_i3,
        "h_2": h_2,
        "CE_i1_h2": CE_i1_h2,
        "Hol_gamma": Hol_gamma,
        "metrics": metrics,
        "theta_AB": theta_AB,
        "V_AB": V_AB,
        "E_V_AB": E_V_AB,
        "E_profit": E_profit,
        "spaces": spaces,
        "arrow_info": arrow_info,
    }


def plot_holonomy(results: dict, save_path: str) -> None:
    """Plot holonomy and payoff for Example 7.2."""
    Hol = results["Hol_gamma"]
    V_AB = results["V_AB"]
    mu_0 = results["spaces"]["t0"].mu

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    states = list(Hol.keys())
    hol_vals = [float(Hol[s]) for s in states]
    probs = [float(mu_0[s]) for s in states]

    axes[0].bar(
        [str(s) for s in states], hol_vals, color=["steelblue", "coral"],
        edgecolor="black", width=0.4,
    )
    axes[0].axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                    label="Hol = 1 (boundary)")
    axes[0].set_xlabel("State ω", fontsize=11)
    axes[0].set_ylabel("Hol(γ)(ω)", fontsize=11)
    axes[0].set_title("Example 7.2: Holonomy per State\n(Stronger AB: Hol ≥ 1 everywhere)", fontsize=11)
    axes[0].legend(fontsize=9)
    for i, (v, p) in enumerate(zip(hol_vals, probs)):
        axes[0].text(i, v + 0.03, f"μ={p:.2f}", ha="center", fontsize=9)

    v_vals = [float(V_AB[s]) for s in states]
    axes[1].bar(
        [str(s) for s in states], v_vals, color=["#aed6f1", "#a9dfbf"],
        edgecolor="black", width=0.4,
    )
    axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="V = 1")
    axes[1].set_xlabel("State ω", fontsize=11)
    axes[1].set_ylabel("V^AB(ω)", fontsize=11)
    axes[1].set_title("Example 7.2: θ^AB Payoff", fontsize=11)
    axes[1].legend(fontsize=9)
    E_text = f"E[V^AB] = {results['E_V_AB']} ≈ {float(results['E_V_AB']):.3f}"
    axes[1].text(
        0.5, 0.95, E_text, transform=axes[1].transAxes,
        ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


if __name__ == "__main__":
    results = run_experiment_2()
    os.makedirs("results", exist_ok=True)
    plot_holonomy(results, "results/exp2_holonomy.png")
    print("\nExperiment 2 complete.")
