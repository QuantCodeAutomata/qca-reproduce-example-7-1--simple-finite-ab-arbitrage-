"""
Experiment 5: Parameter Sensitivity Analysis.

Investigate how holonomy, arbitrage probability, and expected payoff change
as probability weights are varied:
  - Example 7.1: vary μ_2(0) = p over a grid
  - Example 7.2: vary μ_0(0) = q over a grid

Key finding: In both examples, holonomy VALUES are invariant to the parameter;
only the PROBABILITY of exploiting the arbitrage changes.

All computations use exact rational arithmetic.
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
    simplicial_distortion,
    compute_ab_strategy,
    compute_ab_metrics,
)


# ---------------------------------------------------------------------------
# Parametric Example 7.1: vary p = μ_2(0)
# ---------------------------------------------------------------------------

def build_example_71_param(p: Fraction) -> tuple[dict, dict]:
    """
    Build Example 7.1 spaces with μ_2(0) = p (parameter), μ_2(1) = 1-p.

    Parameters
    ----------
    p : Fraction
        Value in (0,1) for μ_2(0).
    """
    assert Fraction(0) < p < Fraction(1), f"p must be in (0,1), got {p}"
    spaces = {
        "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        "t1": ProbSpace("t1", ["*"], {"*": Fraction(1, 1)}),
        "t2": ProbSpace("t2", [0, 1], {0: p, 1: Fraction(1) - p}),
    }
    arrow_info = {
        "i1": {"F_map": {"*": 1}, "source_space": "t1", "target_space": "t0"},
        "i2": {"F_map": {0: "*", 1: "*"}, "source_space": "t2", "target_space": "t1"},
        "i3": {"F_map": {0: 0, 1: 1}, "source_space": "t0", "target_space": "t2"},
    }
    return spaces, arrow_info


def sweep_example_71() -> list[dict]:
    """
    Sweep p = μ_2(0) over k/20 for k=1,...,19 and compute holonomy metrics.

    Analytical result: Hol(γ) = {0:0, 1:4} is INDEPENDENT of p.
    """
    print("\n--- Sweep: Example 7.1 (varying p = μ_2(0)) ---")
    p_grid = [Fraction(k, 20) for k in range(1, 20)]
    # Add paper's original value if not already present
    paper_p = Fraction(1, 4)
    if paper_p not in p_grid:
        p_grid.insert(4, paper_p)

    rows = []
    for p in p_grid:
        spaces, arrow_info = build_example_71_param(p)
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        mu_0 = spaces["t0"].mu
        metrics = compute_ab_metrics(Hol, mu_0)
        _, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol, mu_0)

        # Verify invariance: Hol must always be {0:0, 1:4}
        assert Hol[0] == Fraction(0), f"Hol[0]={Hol[0]} for p={p}"
        assert Hol[1] == Fraction(4), f"Hol[1]={Hol[1]} for p={p}"
        assert metrics["P_AB"] == Fraction(1, 2), f"P_AB={metrics['P_AB']} for p={p}"
        assert E_V_AB == Fraction(5, 2), f"E_V_AB={E_V_AB} for p={p}"

        rows.append({
            "p": p,
            "Hol_0": Hol[0],
            "Hol_1": Hol[1],
            "P_AB": metrics["P_AB"],
            "E_Hol": metrics["E_Hol"],
            "E_V_AB": E_V_AB,
            "E_profit": E_profit,
        })

    print(f"  Swept {len(rows)} values of p.")
    print(f"  Holonomy invariant: Hol = {{0:0, 1:4}} for all p ✓")
    print(f"  P(Hol>1) = 1/2 for all p ✓")
    print(f"  E[V^AB] = 5/2 for all p ✓")
    print(
        "\n  Analytical explanation: The single-point Ω_{t1}={*} collapses all "
        "information.\n  CE(i2, dF_i3) = {*:1} always by normalization; "
        "tail_dist={*:1} always;\n  Hol is determined solely by dF(i1) and "
        "CE(i1,{*:1}) = {0:0,1:2}, independent of p."
    )
    return rows


# ---------------------------------------------------------------------------
# Parametric Example 7.2: vary q = μ_0(0)
# ---------------------------------------------------------------------------

def build_example_72_param(q: Fraction) -> tuple[dict, dict]:
    """
    Build Example 7.2 spaces with μ_0(0) = q (parameter), μ_0(1) = 1-q.
    All other parameters (μ_1, μ_2, maps) are fixed.

    Parameters
    ----------
    q : Fraction
        Value in (0,1) for μ_0(0).
    """
    assert Fraction(0) < q < Fraction(1), f"q must be in (0,1), got {q}"
    spaces = {
        "t0": ProbSpace("t0", [0, 1], {0: q, 1: Fraction(1) - q}),
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


def sweep_example_72() -> list[dict]:
    """
    Sweep q = μ_0(0) over k/20 for k=1,...,19 and compute holonomy metrics.

    Analytical derivation (corrected):
      dF(i3)[u] = q/(1/4) = 4q,  dF(i3)[w] = (1-q)/(1/2) = 2(1-q)
      CE(i2, dF_i3)[a] = 4q,  [b] = 4(1-q),  [c] = 0
      h2[a] = 4q,  h2[b] = 8(1-q),  h2[c] = 0
      CE(i1, h2)[0] = h2[a]*μ_1[a]/μ_0[0] = 4q*(1/4)/q = 1   (cancels q)
      CE(i1, h2)[1] = h2[b]*μ_1[b]/μ_0[1] = 8(1-q)*(1/4)/(1-q) = 2 (cancels 1-q)
      dF(i1)[0] = μ_1[a]/μ_0[0] = (1/4)/q  ← depends on q (≡1 only at q=1/4)
      dF(i1)[1] = (μ_1[b]+μ_1[c])/μ_0[1] = (3/4)/(1-q) ← depends on q

      Hol[0] = dF(i1)[0] * CE(i1,h2)[0] = (1/4)/q * 1 = 1/(4q)
      Hol[1] = dF(i1)[1] * CE(i1,h2)[1] = (3/4)/(1-q) * 2 = 3/(2(1-q))

      INVARIANT: E[Hol] = q*1/(4q) + (1-q)*3/(2(1-q)) = 1/4 + 3/2 = 7/4 (all q)

      Critical threshold q* = 1/4:
        q < 1/4: Hol[0] > 1  → P(Hol>1) = 1  (stronger AB)
        q = 1/4: Hol[0] = 1  → P(Hol>1) = 3/4 (paper's value)
        q > 1/4: Hol[0] < 1  → P(Hol>1) = 1-q (only state 1)
      Hol[1] = 3/(2(1-q)) > 1 for all q ∈ (0,1) (since 3 > 2(1-q) for q > -1/2)
    """
    print("\n--- Sweep: Example 7.2 (varying q = μ_0(0)) ---")
    q_grid = [Fraction(k, 20) for k in range(1, 20)]
    paper_q = Fraction(1, 4)
    if paper_q not in q_grid:
        q_grid.insert(4, paper_q)

    rows = []
    for q in q_grid:
        spaces, arrow_info = build_example_72_param(q)
        Hol = simplicial_distortion(
            ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
        )
        mu_0 = spaces["t0"].mu
        metrics = compute_ab_metrics(Hol, mu_0)
        _, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol, mu_0)

        # Verify correct analytical formulas for Hol
        expected_Hol0 = Fraction(1, 4) / q                    # = 1/(4q)
        expected_Hol1 = Fraction(3, 2) / (Fraction(1) - q)    # = 3/(2(1-q))
        assert Hol[0] == expected_Hol0, (
            f"Hol[0]={Hol[0]} ≠ 1/(4q)={expected_Hol0} at q={q}"
        )
        assert Hol[1] == expected_Hol1, (
            f"Hol[1]={Hol[1]} ≠ 3/(2(1-q))={expected_Hol1} at q={q}"
        )

        # Hol[1] > 1 always (Hol[0] > 1 only when q < 1/4)
        assert Hol[1] > Fraction(1), f"Hol[1]={Hol[1]} should be > 1 for q={q}"

        # E[Hol] = 7/4 is the true invariant
        assert metrics["E_Hol"] == Fraction(7, 4), (
            f"E[Hol]={metrics['E_Hol']} ≠ 7/4 at q={q}"
        )

        # Verify P(Hol>1) formula
        if q < Fraction(1, 4):
            expected_P_AB = Fraction(1)          # both states: stronger AB
        elif q == Fraction(1, 4):
            expected_P_AB = Fraction(3, 4)       # paper's value
        else:
            expected_P_AB = Fraction(1) - q      # only state 1

        assert metrics["P_AB"] == expected_P_AB, (
            f"P_AB={metrics['P_AB']} ≠ expected {expected_P_AB} at q={q}"
        )

        # P(Hol≥1) = 1 only when q ≤ 1/4
        if q <= Fraction(1, 4):
            assert metrics["P_no_loss"] == Fraction(1), (
                f"P(Hol≥1)={metrics['P_no_loss']} ≠ 1 at q={q}"
            )

        rows.append({
            "q": q,
            "Hol_0": Hol[0],
            "Hol_1": Hol[1],
            "P_AB": metrics["P_AB"],
            "P_no_loss": metrics["P_no_loss"],
            "E_Hol": metrics["E_Hol"],
            "E_V_AB": E_V_AB,
            "E_profit": E_profit,
        })

    print(f"  Swept {len(rows)} values of q.")
    print("  Hol[0] = 1/(4q), Hol[1] = 3/(2(1-q)) — depend on q ✓")
    print("  INVARIANT: E[Hol] = 7/4 for all q ✓")
    print("  Critical threshold: q* = 1/4 (paper's value)")
    print("    q < 1/4 → P(Hol>1)=1 (stronger AB)")
    print("    q = 1/4 → P(Hol>1)=3/4 (paper's result)")
    print("    q > 1/4 → P(Hol>1)=1-q (only state 1)")

    # Verify at paper's value q=1/4
    paper_row = next(r for r in rows if r["q"] == Fraction(1, 4))
    assert paper_row["Hol_0"] == Fraction(1), f"Hol[0] at q=1/4: {paper_row['Hol_0']}"
    assert paper_row["Hol_1"] == Fraction(2), f"Hol[1] at q=1/4: {paper_row['Hol_1']}"
    assert paper_row["P_AB"] == Fraction(3, 4), f"P_AB at q=1/4: {paper_row['P_AB']}"
    assert paper_row["E_V_AB"] == Fraction(7, 4), f"E_V_AB at q=1/4: {paper_row['E_V_AB']}"
    print(
        f"\n  At paper's value q=1/4: Hol={{0:1, 1:2}}, P_AB={paper_row['P_AB']}, "
        f"E[V^AB]={paper_row['E_V_AB']} (match paper exactly) ✓"
    )
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity(rows_71: list[dict], rows_72: list[dict], save_path: str) -> None:
    """Plot sensitivity curves for both examples."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Experiment 5: Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")

    # Example 7.1 — Holonomy values vs p (should be flat)
    p_vals = [float(r["p"]) for r in rows_71]
    Hol1_vals = [float(r["Hol_1"]) for r in rows_71]
    Hol0_vals = [float(r["Hol_0"]) for r in rows_71]

    ax = axes[0, 0]
    ax.plot(p_vals, Hol1_vals, "b-o", markersize=4, label="Hol(γ)(ω=1) = 4")
    ax.plot(p_vals, Hol0_vals, "r-s", markersize=4, label="Hol(γ)(ω=0) = 0")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(x=0.25, color="green", linestyle="--", linewidth=1.2, label="Paper value p=1/4")
    ax.set_xlabel("p = μ_2(0)", fontsize=11)
    ax.set_ylabel("Hol(γ)(ω)", fontsize=11)
    ax.set_title("Ex 7.1: Holonomy vs p (INVARIANT)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.3, 4.8)

    # Example 7.1 — E[V^AB] vs p (flat)
    E_vals_71 = [float(r["E_V_AB"]) for r in rows_71]
    ax = axes[0, 1]
    ax.plot(p_vals, E_vals_71, "g-D", markersize=4, label="E[V^AB] = 5/2")
    ax.axhline(y=2.5, color="green", linestyle=":", linewidth=1)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="E=1 (no profit)")
    ax.axvline(x=0.25, color="green", linestyle="--", linewidth=1.2, label="Paper value p=1/4")
    ax.set_xlabel("p = μ_2(0)", fontsize=11)
    ax.set_ylabel("E[V^AB]", fontsize=11)
    ax.set_title("Ex 7.1: E[V^AB] vs p (CONSTANT = 5/2)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 3.2)

    # Example 7.2 — Hol values vs q (nonlinear: Hol[0]=1/(4q), Hol[1]=3/(2(1-q)))
    q_vals = [float(r["q"]) for r in rows_72]
    Hol0_72 = [float(r["Hol_0"]) for r in rows_72]
    Hol1_72 = [float(r["Hol_1"]) for r in rows_72]
    P_AB_vals = [float(r["P_AB"]) for r in rows_72]

    ax = axes[1, 0]
    ax.plot(q_vals, Hol0_72, "b-o", markersize=4, label="Hol[0] = 1/(4q)")
    ax.plot(q_vals, Hol1_72, "r-s", markersize=4, label="Hol[1] = 3/(2(1−q))")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, label="Hol=1")
    ax.axvline(x=0.25, color="green", linestyle="--", linewidth=1.2, label="Paper q=1/4")
    ax.set_xlabel("q = μ_0(0)", fontsize=11)
    ax.set_ylabel("Hol(γ)(ω)", fontsize=11)
    ax.set_title("Ex 7.2: Holonomy vs q\n(critical threshold at q*=1/4)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(10, max(max(Hol0_72), max(Hol1_72)) * 1.1))

    # Example 7.2 — P(Hol>1) and E[Hol] vs q
    E_Hol_vals = [float(r["E_Hol"]) for r in rows_72]
    ax = axes[1, 1]
    ax.plot(q_vals, P_AB_vals, "b-o", markersize=4, label="P(Hol>1)")
    ax.plot(q_vals, E_Hol_vals, "r-s", markersize=4, label="E[Hol] = 7/4 (invariant)")
    ax.axhline(y=7 / 4, color="red", linestyle=":", linewidth=1.5)
    ax.axvline(x=0.25, color="green", linestyle="--", linewidth=1.2, label="Paper q=1/4")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("q = μ_0(0)", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("Ex 7.2: P(Hol>1) and E[Hol] vs q\n(E[Hol]=7/4 constant)", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def run_experiment_5() -> dict:
    """Run full sensitivity analysis for Experiments 7.1 and 7.2."""
    print("=" * 60)
    print("EXPERIMENT 5: Parameter Sensitivity Analysis")
    print("=" * 60)

    rows_71 = sweep_example_71()
    rows_72 = sweep_example_72()

    df71 = pd.DataFrame(rows_71)
    df72 = pd.DataFrame(rows_72)

    print("\nExample 7.1 sweep (first 5 rows):")
    print(df71.head().to_string(index=False))
    print("\nExample 7.2 sweep (first 5 rows):")
    print(df72.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("Key Conclusions:")
    print("  Example 7.1 (vary p = μ_2(0)):  Hol={0:0, 1:4} INVARIANT for all p.")
    print("    The single-point Ω_t1={*} collapses information; CE normalises away p.")
    print("  Example 7.2 (vary q = μ_0(0)):  Hol values DEPEND on q:")
    print("    Hol[0] = 1/(4q),  Hol[1] = 3/(2(1-q))")
    print("    But E[Hol] = 7/4 is INVARIANT (expected holonomy constant).")
    print("    Critical threshold q*=1/4 separates stronger from simple AB arbitrage.")
    print("    Paper's Examples use q=1/4 → Hol={0:1, 1:2} (the measure-preserving point).")
    print("=" * 60)

    return {"rows_71": rows_71, "rows_72": rows_72}


if __name__ == "__main__":
    results = run_experiment_5()
    os.makedirs("results", exist_ok=True)
    plot_sensitivity(results["rows_71"], results["rows_72"], "results/exp5_sensitivity.png")
    print("\nExperiment 5 complete.")
