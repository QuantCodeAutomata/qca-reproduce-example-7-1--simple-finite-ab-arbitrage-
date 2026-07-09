"""
Experiment 1: Reproduce Example 7.1 — Simple Finite AB Arbitrage with Hol(γ) = {0:0, 1:4}.

Three finite probability spaces connected by measurable maps forming a triangular
loop in time category T = {t_0, t_1, t_2}. All computations use exact rational
arithmetic (fractions.Fraction).

Expected holonomy: Hol(γ) = {0: Fraction(0,1), 1: Fraction(4,1)}
Expected E[V^AB] = 5/2, E[profit] = 3/2
"""

from fractions import Fraction
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
# Define Input Structures (Phase 1)
# ---------------------------------------------------------------------------

def build_example_71() -> tuple[dict, dict]:
    """
    Build probability spaces and arrow_info for Example 7.1.

    Ω_{t0} = {0,1},  μ_0 = {0:1/2, 1:1/2}
    Ω_{t1} = {*},    μ_1 = {*:1}
    Ω_{t2} = {0,1},  μ_2 = {0:1/4, 1:3/4}

    F(i1): Ω_{t1}→Ω_{t0}: *↦1
    F(i2): Ω_{t2}→Ω_{t1}: 0↦*, 1↦*
    F(i3): Ω_{t0}→Ω_{t2}: 0↦0, 1↦1  (identity)
    """
    spaces = {
        "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        "t1": ProbSpace("t1", ["*"], {"*": Fraction(1, 1)}),
        "t2": ProbSpace("t2", [0, 1], {0: Fraction(1, 4), 1: Fraction(3, 4)}),
    }

    # source_space = T^op-source (Ω FROM which F(i_k) maps)
    # target_space = T^op-target (Ω TO which F(i_k) maps)
    arrow_info = {
        "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
        "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
        "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
    }
    return spaces, arrow_info


# ---------------------------------------------------------------------------
# Step-by-step Computation with Intermediate Verification (Phase 2–4)
# ---------------------------------------------------------------------------

def run_experiment_1() -> dict:
    """
    Execute all phases of Experiment 1 and return results dict.

    Returns
    -------
    dict
        Contains all intermediate and final results.
    """
    spaces, arrow_info = build_example_71()
    mu_0 = spaces["t0"].mu
    mu_1 = spaces["t1"].mu
    mu_2 = spaces["t2"].mu

    print("=" * 60)
    print("EXPERIMENT 1: Example 7.1 — AB Arbitrage with Hol = {0:0, 1:4}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step A: dF(i3) on Ω_{t2} = {0,1}
    # ----------------------------------------------------------------
    dF_i3 = distortion("i3", spaces, arrow_info)
    print("\nStep A: dF(i3) on Ω_t2 =", dF_i3)
    assert dF_i3 == {0: Fraction(2, 1), 1: Fraction(2, 3)}, f"dF_i3 mismatch: {dF_i3}"

    # ----------------------------------------------------------------
    # Step B: CE(i2, dF_i3) on Ω_{t1} = {*}
    # ----------------------------------------------------------------
    CE_i2_dF_i3 = conditional_expectation("i2", dF_i3, spaces, arrow_info)
    print("Step B: CE(i2, dF_i3) =", CE_i2_dF_i3)
    assert CE_i2_dF_i3 == {"*": Fraction(1, 1)}, f"CE_i2_dF_i3 mismatch: {CE_i2_dF_i3}"

    # ----------------------------------------------------------------
    # Step C: dF(i2) on Ω_{t1} = {*}
    # ----------------------------------------------------------------
    dF_i2 = distortion("i2", spaces, arrow_info)
    print("Step C: dF(i2) =", dF_i2)
    assert dF_i2 == {"*": Fraction(1, 1)}, f"dF_i2 mismatch: {dF_i2}"

    # ----------------------------------------------------------------
    # Step D: tail_distortion = dF(i2) * CE(i2, dF_i3) on Ω_{t1}
    # ----------------------------------------------------------------
    tail_dist = pointwise_product(dF_i2, CE_i2_dF_i3)
    print("Step D: tail_distortion =", tail_dist)
    assert tail_dist == {"*": Fraction(1, 1)}

    # ----------------------------------------------------------------
    # Step E: CE(i1, tail_dist) on Ω_{t0}
    # ----------------------------------------------------------------
    CE_i1_tail = conditional_expectation("i1", tail_dist, spaces, arrow_info)
    print("Step E: CE(i1, tail_distortion) =", CE_i1_tail)
    assert CE_i1_tail == {0: Fraction(0, 1), 1: Fraction(2, 1)}, (
        f"CE_i1_tail mismatch: {CE_i1_tail}"
    )

    # ----------------------------------------------------------------
    # Step F: dF(i1) on Ω_{t0}
    # ----------------------------------------------------------------
    dF_i1 = distortion("i1", spaces, arrow_info)
    print("Step F: dF(i1) =", dF_i1)
    assert dF_i1 == {0: Fraction(0, 1), 1: Fraction(2, 1)}, f"dF_i1 mismatch: {dF_i1}"

    # ----------------------------------------------------------------
    # Step G: Hol(γ) = dF(i1) * CE(i1, tail_dist)
    # ----------------------------------------------------------------
    Hol_manual = pointwise_product(dF_i1, CE_i1_tail)
    print("Step G: Hol(γ) [manual] =", Hol_manual)
    assert Hol_manual == {0: Fraction(0, 1), 1: Fraction(4, 1)}, (
        f"Holonomy mismatch: {Hol_manual}"
    )

    # ----------------------------------------------------------------
    # Phase 3: Full recursive computation
    # ----------------------------------------------------------------
    sigma_loop = ["t0", "t1", "t2", "t0"]
    arrows_loop = ["i1", "i2", "i3"]
    Hol_gamma = simplicial_distortion(sigma_loop, arrows_loop, spaces, arrow_info)
    print("\nPhase 3: Hol(γ) [recursive] =", Hol_gamma)
    assert Hol_gamma == {0: Fraction(0, 1), 1: Fraction(4, 1)}, (
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

    assert metrics["P_AB"] == Fraction(1, 2), f"P_AB = {metrics['P_AB']}"
    assert metrics["P_no_loss"] == Fraction(1, 2), f"P_no_loss = {metrics['P_no_loss']}"
    assert metrics["P_weak_AB"] == Fraction(1, 1), f"P_weak_AB = {metrics['P_weak_AB']}"
    assert metrics["E_Hol"] == Fraction(2, 1), f"E_Hol = {metrics['E_Hol']}"
    assert metrics["P_AB"] > Fraction(0), "P(Hol>1) > 0 required for AB arbitrage"

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
    assert V_AB == {0: Fraction(1, 1), 1: Fraction(4, 1)}, f"V_AB = {V_AB}"
    assert E_V_AB == Fraction(5, 2), f"E_V_AB = {E_V_AB}"
    assert E_profit == Fraction(3, 2), f"E_profit = {E_profit}"

    # ----------------------------------------------------------------
    # Phase 7: Weak AB Strategy
    # ----------------------------------------------------------------
    print("\nPhase 7: Weak AB Strategy")
    for eps in [Fraction(0), Fraction(1, 10), Fraction(1, 2)]:
        theta_wAB, V_wAB = compute_weak_ab_strategy(Hol_gamma, mu_0, eps)
        print(f"  ε={eps}: θ^wAB={theta_wAB}, V^wAB={V_wAB}")
        # At ω=0: Hol=0 < 1-ε, so θ^wAB(0)=-1 (inadmissible reverse execution)
        assert theta_wAB[0] == Fraction(-1), (
            f"At ω=0 (Hol=0 < 1-ε={1-eps}), θ^wAB should be -1 but got {theta_wAB[0]}"
        )
        assert theta_wAB[1] == Fraction(1), (
            f"At ω=1 (Hol=4 > 1+ε), θ^wAB should be +1"
        )
        # Note: reverse execution at ω=0 is inadmissible (Hol(γ)(0)=0)
        print(f"    INADMISSIBILITY at ω=0: Hol(γ)(0)=0 → reverse execution undefined (1/0)")

    # ----------------------------------------------------------------
    # Phase 8: Validation Checks
    # ----------------------------------------------------------------
    print("\nPhase 8: Validation Checks")

    # Check 1: Distortion normalization
    norm_i1 = check_normalization(dF_i1, mu_0)
    norm_i2 = check_normalization(dF_i2, mu_1)
    norm_i3 = check_normalization(dF_i3, mu_2)
    print(f"  Σ dF(i1)·μ_t0 = {norm_i1} {'✓' if norm_i1 == 1 else '✗'}")
    print(f"  Σ dF(i2)·μ_t1 = {norm_i2} {'✓' if norm_i2 == 1 else '✗'}")
    print(f"  Σ dF(i3)·μ_t2 = {norm_i3} {'✓' if norm_i3 == 1 else '✗'}")
    assert norm_i1 == Fraction(1), f"norm_i1={norm_i1}"
    assert norm_i2 == Fraction(1), f"norm_i2={norm_i2}"
    assert norm_i3 == Fraction(1), f"norm_i3={norm_i3}"

    # Check 2: CE integral preservation — use ones on source space so that
    # LHS = Σ dF(i_k)[y]*μ_target[y] = 1 and RHS = Σ 1*μ_source[x] = 1
    for arrow in ["i1", "i2", "i3"]:
        source_sp = spaces[arrow_info[arrow]["source_space"]]
        ones_f = {x: Fraction(1) for x in source_sp.omega}
        lhs, rhs = check_ce_integral_preservation(arrow, ones_f, spaces, arrow_info)
        print(f"  CE integral preservation {arrow}: {lhs} == {rhs} {'✓' if lhs == rhs else '✗'}")
        assert lhs == rhs, f"CE integral preservation failed for {arrow}"

    # Check 3: Payoff non-negativity
    assert all(V_AB[w] >= Fraction(1) for w in Hol_gamma), "V^AB < 1 at some state"
    print("  V^AB(ω) >= 1 everywhere ✓")

    # Check 4: Profit when Hol > 1
    for w in Hol_gamma:
        if Hol_gamma[w] > Fraction(1):
            assert V_AB[w] > Fraction(1), f"V^AB({w}) not > 1 when Hol={Hol_gamma[w]}"
    print("  V^AB(ω) > 1 on {Hol > 1} ✓")

    print("\nAll assertions PASSED for Experiment 1 ✓")

    return {
        "dF_i1": dF_i1,
        "dF_i2": dF_i2,
        "dF_i3": dF_i3,
        "CE_i2_dF_i3": CE_i2_dF_i3,
        "tail_dist": tail_dist,
        "CE_i1_tail": CE_i1_tail,
        "Hol_gamma": Hol_gamma,
        "metrics": metrics,
        "theta_AB": theta_AB,
        "V_AB": V_AB,
        "E_V_AB": E_V_AB,
        "E_profit": E_profit,
        "spaces": spaces,
        "arrow_info": arrow_info,
    }


# ---------------------------------------------------------------------------
# Visualization & Reporting
# ---------------------------------------------------------------------------

def make_tables(results: dict) -> pd.DataFrame:
    """Build a summary DataFrame for distortion and holonomy values."""
    rows = []
    rows.append({"quantity": "dF(i1)[0]", "value": str(results["dF_i1"][0]),
                 "domain": "Ω_t0"})
    rows.append({"quantity": "dF(i1)[1]", "value": str(results["dF_i1"][1]),
                 "domain": "Ω_t0"})
    rows.append({"quantity": "dF(i2)[*]", "value": str(results["dF_i2"]["*"]),
                 "domain": "Ω_t1"})
    rows.append({"quantity": "dF(i3)[0]", "value": str(results["dF_i3"][0]),
                 "domain": "Ω_t2"})
    rows.append({"quantity": "dF(i3)[1]", "value": str(results["dF_i3"][1]),
                 "domain": "Ω_t2"})
    rows.append({"quantity": "Hol(γ)[0]", "value": str(results["Hol_gamma"][0]),
                 "domain": "Ω_t0"})
    rows.append({"quantity": "Hol(γ)[1]", "value": str(results["Hol_gamma"][1]),
                 "domain": "Ω_t0"})
    rows.append({"quantity": "E[V^AB]", "value": str(results["E_V_AB"]),
                 "domain": "-"})
    rows.append({"quantity": "E[V^AB]-1", "value": str(results["E_profit"]),
                 "domain": "-"})
    return pd.DataFrame(rows)


def plot_holonomy(results: dict, save_path: str) -> None:
    """Plot holonomy distribution and trading strategy payoff for Example 7.1."""
    Hol = results["Hol_gamma"]
    V_AB = results["V_AB"]
    mu_0 = results["spaces"]["t0"].mu

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Holonomy bar chart
    states = list(Hol.keys())
    hol_vals = [float(Hol[s]) for s in states]
    probs = [float(mu_0[s]) for s in states]

    axes[0].bar(
        [str(s) for s in states], hol_vals, color=["steelblue", "coral"],
        edgecolor="black", width=0.4,
    )
    axes[0].axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                    label="Hol = 1 (no arbitrage)")
    axes[0].set_xlabel("State ω", fontsize=11)
    axes[0].set_ylabel("Hol(γ)(ω)", fontsize=11)
    axes[0].set_title("Example 7.1: Holonomy per State", fontsize=12)
    axes[0].legend(fontsize=9)
    for i, (v, p) in enumerate(zip(hol_vals, probs)):
        axes[0].text(i, v + 0.1, f"μ={p:.2f}", ha="center", fontsize=9)

    # Right: Trading strategy payoff
    v_vals = [float(V_AB[s]) for s in states]
    axes[1].bar(
        [str(s) for s in states], v_vals, color=["#aed6f1", "#a9dfbf"],
        edgecolor="black", width=0.4,
    )
    axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1.5,
                    label="V = 1 (no profit)")
    axes[1].set_xlabel("State ω", fontsize=11)
    axes[1].set_ylabel("V^AB(ω)", fontsize=11)
    axes[1].set_title("Example 7.1: θ^AB Payoff", fontsize=12)
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
    results = run_experiment_1()

    df_table = make_tables(results)
    print("\nSummary Table:")
    print(df_table.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    plot_holonomy(results, "results/exp1_holonomy.png")
    print("\nExperiment 1 complete.")
