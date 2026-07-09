"""
Experiment 6: Extended Loop Holonomy.

(A) 4-arrow loop in T_4 = {t0,t1,t2,t3} — verifies recursive formula for n=4.
(B) Two loops sharing a common first arrow — one trivial, one non-trivial.
(C) Foreign exchange (FX) triangle analog — stochastic triangular arbitrage.

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
    conditional_expectation,
    distortion,
    pointwise_product,
    simplicial_distortion,
    compute_ab_strategy,
    compute_ab_metrics,
    check_normalization,
)


# ===========================================================================
# Sub-experiment A: 4-Arrow Loop
# ===========================================================================

def build_4arrow_loop() -> tuple[dict, dict]:
    """
    Build spaces and arrow_info for the 4-arrow loop in T_4.

    Ω_{t0}={0,1},   μ_0={0:1/2, 1:1/2}
    Ω_{t1}={a,b},   μ_1={a:1/4, b:3/4}
    Ω_{t2}={x,y,z}, μ_2={x:1/3, y:1/3, z:1/3}
    Ω_{t3}={p,q},   μ_3={p:2/3, q:1/3}

    F(i1): {a:0, b:0}   — both a,b map to 0
    F(i2): {x:a, y:a, z:b}
    F(i3): {p:x, q:z}
    F(i4): {0:p, 1:q}
    """
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
    return spaces, arrow_info


def run_subexp_a() -> dict:
    """Run 4-arrow loop holonomy computation."""
    print("\n--- Sub-experiment A: 4-Arrow Loop ---")
    spaces, arrow_info = build_4arrow_loop()
    mu_0 = spaces["t0"].mu

    # Individual distortions
    dF_i1 = distortion("i1", spaces, arrow_info)
    dF_i2 = distortion("i2", spaces, arrow_info)
    dF_i3 = distortion("i3", spaces, arrow_info)
    dF_i4 = distortion("i4", spaces, arrow_info)
    print(f"  dF(i1) = {dF_i1}")
    print(f"  dF(i2) = {dF_i2}")
    print(f"  dF(i3) = {dF_i3}")
    print(f"  dF(i4) = {dF_i4}")

    # Expected values (from methodology)
    assert dF_i1 == {0: Fraction(2, 1), 1: Fraction(0, 1)}, f"dF_i1: {dF_i1}"
    assert dF_i2 == {"a": Fraction(8, 3), "b": Fraction(4, 9)}, f"dF_i2: {dF_i2}"
    assert dF_i3 == {"x": Fraction(2, 1), "y": Fraction(0, 1), "z": Fraction(1, 1)}, (
        f"dF_i3: {dF_i3}"
    )
    assert dF_i4 == {"p": Fraction(3, 4), "q": Fraction(3, 2)}, f"dF_i4: {dF_i4}"
    print("  All individual distortions match expected values ✓")

    # Manual recursive computation (expanded 4-level formula)
    # Level 4: h4 = dF(i4) on Ω_{t3}
    h4 = dF_i4
    print(f"\n  h4 = dF(i4) = {h4}")

    # Level 3: h3 = dF(i3) * CE(i3, h4) on Ω_{t2}
    CE_i3_h4 = conditional_expectation("i3", h4, spaces, arrow_info)
    print(f"  CE(i3, h4) = {CE_i3_h4}")
    assert CE_i3_h4 == {
        "x": Fraction(3, 2), "y": Fraction(0, 1), "z": Fraction(3, 2)
    }, f"CE_i3_h4: {CE_i3_h4}"
    h3 = pointwise_product(dF_i3, CE_i3_h4)
    print(f"  h3 = dF(i3) * CE(i3,h4) = {h3}")
    assert h3 == {"x": Fraction(3, 1), "y": Fraction(0, 1), "z": Fraction(3, 2)}, (
        f"h3: {h3}"
    )

    # Level 2: h2 = dF(i2) * CE(i2, h3) on Ω_{t1}
    CE_i2_h3 = conditional_expectation("i2", h3, spaces, arrow_info)
    print(f"  CE(i2, h3) = {CE_i2_h3}")
    assert CE_i2_h3 == {"a": Fraction(4, 1), "b": Fraction(2, 3)}, (
        f"CE_i2_h3: {CE_i2_h3}"
    )
    h2 = pointwise_product(dF_i2, CE_i2_h3)
    print(f"  h2 = dF(i2) * CE(i2,h3) = {h2}")
    assert h2 == {"a": Fraction(32, 3), "b": Fraction(8, 27)}, f"h2: {h2}"

    # Level 1: Hol = dF(i1) * CE(i1, h2) on Ω_{t0}
    CE_i1_h2 = conditional_expectation("i1", h2, spaces, arrow_info)
    print(f"  CE(i1, h2) = {CE_i1_h2}")
    assert CE_i1_h2 == {0: Fraction(52, 9), 1: Fraction(0, 1)}, (
        f"CE_i1_h2: {CE_i1_h2}"
    )
    Hol_manual = pointwise_product(dF_i1, CE_i1_h2)
    print(f"  Hol(γ4) [manual] = {Hol_manual}")
    assert Hol_manual == {0: Fraction(104, 9), 1: Fraction(0, 1)}, (
        f"Hol_manual: {Hol_manual}"
    )

    # Verify via recursive simplicial_distortion
    Hol_recursive = simplicial_distortion(
        ["t0", "t1", "t2", "t3", "t0"], ["i1", "i2", "i3", "i4"], spaces, arrow_info
    )
    print(f"  Hol(γ4) [recursive] = {Hol_recursive}")
    assert Hol_recursive == Hol_manual, (
        f"Manual vs recursive mismatch:\n  manual={Hol_manual}\n  recursive={Hol_recursive}"
    )
    print("  Manual == Recursive ✓")

    # Metrics
    metrics = compute_ab_metrics(Hol_recursive, mu_0)
    _, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol_recursive, mu_0)
    print(f"\n  P(Hol>1) = {metrics['P_AB']}")
    print(f"  E[Hol] = {metrics['E_Hol']}")
    print(f"  E[V^AB] = {E_V_AB}")
    print(f"  E[profit] = {E_profit}")

    assert metrics["P_AB"] == Fraction(1, 2)  # only state 0 has Hol=104/9>1
    assert all(V_AB[w] >= Fraction(1) for w in Hol_recursive)

    # Normalization checks
    for k in ["i1", "i2", "i3", "i4"]:
        target_name = arrow_info[k]["target_space"]
        mu_target = spaces[target_name].mu
        dF_k = distortion(k, spaces, arrow_info)
        norm = check_normalization(dF_k, mu_target)
        assert norm == Fraction(1), f"dF({k}) normalization = {norm} ≠ 1"
    print("  All normalization checks pass ✓")
    print("  Sub-experiment A COMPLETE ✓")

    return {
        "Hol": Hol_recursive,
        "metrics": metrics,
        "E_V_AB": E_V_AB,
        "E_profit": E_profit,
        "spaces": spaces,
        "arrow_info": arrow_info,
    }


# ===========================================================================
# Sub-experiment B: Two Loops Sharing First Arrow
# ===========================================================================

def build_two_loops() -> tuple[dict, dict]:
    """
    Build spaces and arrow_info for two loops sharing arrow i1.

    Loop γ1 = (t0,t1,t2,t0): arrows (i1,i2,i3) — non-trivial holonomy
    Loop γ2 = (t0,t1,t3,t0): arrows (i1,i4,i5) — trivial holonomy

    Ω_{t0}={0,1},  μ_0={0:1/2, 1:1/2}
    Ω_{t1}={a,b},  μ_1={a:1/2, b:1/2}
    Ω_{t2}={u,v},  μ_2={u:1/3, v:2/3}
    Ω_{t3}={s,t_}, μ_3={s:1/2, t_:1/2}

    F(i1): {a:0, b:1}        — measure-preserving
    F(i2): {u:a, v:a}        — both map to a (NOT measure-preserving)
    F(i3): {0:u, 1:v}        — measure-distorting
    F(i4): {s:a, t_:b}       — measure-preserving (μ_3→μ_1)
    F(i5): {0:s, 1:t_}       — measure-preserving (μ_0→μ_3)
    """
    spaces = {
        "t0": ProbSpace("t0", [0, 1], {0: Fraction(1, 2), 1: Fraction(1, 2)}),
        "t1": ProbSpace("t1", ["a", "b"], {"a": Fraction(1, 2), "b": Fraction(1, 2)}),
        "t2": ProbSpace("t2", ["u", "v"], {"u": Fraction(1, 3), "v": Fraction(2, 3)}),
        "t3_": ProbSpace("t3_", ["s", "t_"], {"s": Fraction(1, 2), "t_": Fraction(1, 2)}),
    }
    arrow_info = {
        "i1": {"F_map": {"a": 0, "b": 1}, "source_space": "t1", "target_space": "t0"},
        "i2": {"F_map": {"u": "a", "v": "a"}, "source_space": "t2", "target_space": "t1"},
        "i3": {"F_map": {0: "u", 1: "v"}, "source_space": "t0", "target_space": "t2"},
        "i4": {"F_map": {"s": "a", "t_": "b"}, "source_space": "t3_", "target_space": "t1"},
        "i5": {"F_map": {0: "s", 1: "t_"}, "source_space": "t0", "target_space": "t3_"},
    }
    return spaces, arrow_info


def run_subexp_b() -> dict:
    """Run two-loop comparison."""
    print("\n--- Sub-experiment B: Two Loops Sharing First Arrow ---")
    spaces, arrow_info = build_two_loops()
    mu_0 = spaces["t0"].mu

    # Loop γ1 = (i1, i2, i3)
    Hol_1 = simplicial_distortion(
        ["t0", "t1", "t2", "t0"], ["i1", "i2", "i3"], spaces, arrow_info
    )
    print(f"  Hol(γ1) = {Hol_1}")
    assert Hol_1 == {0: Fraction(4, 1), 1: Fraction(0, 1)}, f"Hol_1: {Hol_1}"

    metrics_1 = compute_ab_metrics(Hol_1, mu_0)
    _, V_AB_1, E_V_AB_1, _ = compute_ab_strategy(Hol_1, mu_0)
    print(f"  P(Hol1>1) = {metrics_1['P_AB']}  → AB arbitrage ✓")
    assert metrics_1["P_AB"] == Fraction(1, 2)

    # Loop γ2 = (i1, i4, i5)
    Hol_2 = simplicial_distortion(
        ["t0", "t1", "t3_", "t0"], ["i1", "i4", "i5"], spaces, arrow_info
    )
    print(f"\n  Hol(γ2) = {Hol_2}")
    assert Hol_2 == {0: Fraction(1, 1), 1: Fraction(1, 1)}, f"Hol_2: {Hol_2}"

    metrics_2 = compute_ab_metrics(Hol_2, mu_0)
    print(f"  P(Hol2>1) = {metrics_2['P_AB']}  → NO AB arbitrage ✓")
    assert metrics_2["P_AB"] == Fraction(0)

    print("\n  COMPARISON:")
    print(f"    Loop γ1: Hol={Hol_1}, P(AB)={metrics_1['P_AB']} → ARBITRAGE")
    print(f"    Loop γ2: Hol={Hol_2}, P(AB)={metrics_2['P_AB']} → NO ARBITRAGE")
    print("  A market can have loops with trivial AND non-trivial holonomy ✓")
    print("  Sub-experiment B COMPLETE ✓")

    return {
        "Hol_1": Hol_1, "metrics_1": metrics_1,
        "Hol_2": Hol_2, "metrics_2": metrics_2,
        "spaces": spaces, "arrow_info": arrow_info,
    }


# ===========================================================================
# Sub-experiment C: Foreign Exchange Triangle Analog
# ===========================================================================

def build_fx_triangle() -> tuple[dict, dict]:
    """
    Build FX triangle: USD → EUR → GBP → USD loop.

    Ω_USD = {H, L},  μ_USD = {H:1/2, L:1/2}
    Ω_EUR = {h, l},  μ_EUR = {h:1/3, l:2/3}
    Ω_GBP = {hi,lo}, μ_GBP = {hi:1/4, lo:3/4}

    F_EURUSD: Ω_EUR→Ω_USD: h↦H, l↦H   (both EUR states → USD High)
    F_GBPEUR: Ω_GBP→Ω_EUR: hi↦h, lo↦l (measure-distorting)
    F_USDGBP: Ω_USD→Ω_GBP: H↦hi, L↦lo (measure-distorting)
    """
    spaces = {
        "USD": ProbSpace("USD", ["H", "L"], {"H": Fraction(1, 2), "L": Fraction(1, 2)}),
        "EUR": ProbSpace("EUR", ["h", "l"], {"h": Fraction(1, 3), "l": Fraction(2, 3)}),
        "GBP": ProbSpace("GBP", ["hi", "lo"], {"hi": Fraction(1, 4), "lo": Fraction(3, 4)}),
    }
    arrow_info = {
        "EURUSD": {
            "F_map": {"h": "H", "l": "H"},
            "source_space": "EUR", "target_space": "USD",
        },
        "GBPEUR": {
            "F_map": {"hi": "h", "lo": "l"},
            "source_space": "GBP", "target_space": "EUR",
        },
        "USDGBP": {
            "F_map": {"H": "hi", "L": "lo"},
            "source_space": "USD", "target_space": "GBP",
        },
    }
    return spaces, arrow_info


def run_subexp_c() -> dict:
    """Run FX triangle holonomy."""
    print("\n--- Sub-experiment C: FX Triangle Analog ---")
    spaces, arrow_info = build_fx_triangle()
    mu_USD = spaces["USD"].mu
    mu_EUR = spaces["EUR"].mu
    mu_GBP = spaces["GBP"].mu

    dF_EURUSD = distortion("EURUSD", spaces, arrow_info)
    dF_GBPEUR = distortion("GBPEUR", spaces, arrow_info)
    dF_USDGBP = distortion("USDGBP", spaces, arrow_info)
    print(f"  dF(EURUSD) on Ω_USD = {dF_EURUSD}")
    print(f"  dF(GBPEUR) on Ω_EUR = {dF_GBPEUR}")
    print(f"  dF(USDGBP) on Ω_GBP = {dF_USDGBP}")

    # Normalization checks
    norm_eu = check_normalization(dF_EURUSD, mu_USD)
    norm_ge = check_normalization(dF_GBPEUR, mu_EUR)
    norm_ug = check_normalization(dF_USDGBP, mu_GBP)
    assert norm_eu == Fraction(1), f"dF_EURUSD norm = {norm_eu}"
    assert norm_ge == Fraction(1), f"dF_GBPEUR norm = {norm_ge}"
    assert norm_ug == Fraction(1), f"dF_USDGBP norm = {norm_ug}"
    print("  All distortions normalized ✓")

    # Loop: USD→EUR→GBP→USD
    # In T: EURUSD: USD→EUR, GBPEUR: EUR→GBP, USDGBP: GBP→USD
    # sigma_objects starts and ends at USD
    Hol_FX = simplicial_distortion(
        ["USD", "EUR", "GBP", "USD"],
        ["EURUSD", "GBPEUR", "USDGBP"],
        spaces, arrow_info,
    )
    print(f"\n  Hol(γ_FX) = {Hol_FX}")

    metrics = compute_ab_metrics(Hol_FX, mu_USD)
    _, V_AB, E_V_AB, E_profit = compute_ab_strategy(Hol_FX, mu_USD)
    print(f"  P(Hol>1) = {metrics['P_AB']}")
    print(f"  E[Hol] = {metrics['E_Hol']}")
    print(f"  E[V^AB] = {E_V_AB}, E[profit] = {E_profit}")
    print(f"  θ^AB = {dict((k, str(v)) for k, v in zip(['H','L'], V_AB.values()))}")

    if metrics["P_AB"] > Fraction(0):
        print("  ✓ FX triangular arbitrage detected (P(Hol>1) > 0)")
    else:
        print("  FX loop has no AB arbitrage (P(Hol>1) = 0)")

    assert all(V_AB[w] >= Fraction(1) for w in Hol_FX), "V^AB < 1 in FX triangle"
    print("  V^AB ≥ 1 everywhere ✓")
    print("  Sub-experiment C COMPLETE ✓")

    return {
        "Hol_FX": Hol_FX,
        "metrics": metrics,
        "E_V_AB": E_V_AB,
        "E_profit": E_profit,
        "spaces": spaces,
        "arrow_info": arrow_info,
    }


# ===========================================================================
# Full Experiment 6 Runner
# ===========================================================================

def run_experiment_6() -> dict:
    """Run all three sub-experiments of Experiment 6."""
    print("=" * 60)
    print("EXPERIMENT 6: Extended Loop Holonomy")
    print("=" * 60)

    r_a = run_subexp_a()
    r_b = run_subexp_b()
    r_c = run_subexp_c()

    print("\n" + "=" * 60)
    print("EXPERIMENT 6 SUMMARY:")
    print(f"  (A) 4-arrow loop: Hol={dict(r_a['Hol'])}, E[profit]={r_a['E_profit']}")
    print(f"  (B) Two loops: Hol_γ1={dict(r_b['Hol_1'])}, Hol_γ2={dict(r_b['Hol_2'])}")
    print(f"  (C) FX triangle: Hol={dict(r_c['Hol_FX'])}, E[profit]={r_c['E_profit']}")
    print("=" * 60)

    return {"A": r_a, "B": r_b, "C": r_c}


def plot_exp6(results: dict, save_path: str) -> None:
    """Plot holonomy bar charts for all three sub-experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Experiment 6: Extended Loop Holonomy", fontsize=13, fontweight="bold")

    # Sub-experiment A
    Hol_A = results["A"]["Hol"]
    ax = axes[0]
    states_a = list(Hol_A.keys())
    ax.bar([str(s) for s in states_a], [float(Hol_A[s]) for s in states_a],
           color=["coral", "steelblue"], edgecolor="black", width=0.4)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Hol=1")
    ax.set_title(f"(A) 4-Arrow Loop\nHol={{{0}:{float(Hol_A[0]):.2f}, {1}:{float(Hol_A[1]):.2f}}}", fontsize=10)
    ax.set_xlabel("State ω"); ax.set_ylabel("Hol(γ4)(ω)")
    ax.legend(fontsize=8)

    # Sub-experiment B
    Hol_1 = results["B"]["Hol_1"]
    Hol_2 = results["B"]["Hol_2"]
    ax = axes[1]
    st1 = list(Hol_1.keys())
    width = 0.3
    x = range(len(st1))
    ax.bar([i - width / 2 for i in x], [float(Hol_1[s]) for s in st1],
           width=width, color="coral", edgecolor="black", label="Loop γ1")
    ax.bar([i + width / 2 for i in x], [float(Hol_2[s]) for s in st1],
           width=width, color="lightgreen", edgecolor="black", label="Loop γ2")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xticks(list(x)); ax.set_xticklabels([str(s) for s in st1])
    ax.set_title("(B) Two Loops: γ1 (AB) vs γ2 (trivial)", fontsize=10)
    ax.set_xlabel("State ω"); ax.set_ylabel("Hol(γ)(ω)")
    ax.legend(fontsize=8)

    # Sub-experiment C
    Hol_C = results["C"]["Hol_FX"]
    ax = axes[2]
    states_c = list(Hol_C.keys())
    ax.bar([str(s) for s in states_c], [float(Hol_C[s]) for s in states_c],
           color=["gold", "skyblue"], edgecolor="black", width=0.4)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Hol=1")
    ax.set_title("(C) FX Triangle Loop", fontsize=10)
    ax.set_xlabel("USD State"); ax.set_ylabel("Hol(γ_FX)(ω)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


if __name__ == "__main__":
    results = run_experiment_6()
    os.makedirs("results", exist_ok=True)
    plot_exp6(results, "results/exp6_extended_loops.png")
    print("\nExperiment 6 complete.")
