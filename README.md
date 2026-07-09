# AB Arbitrage via Categorical Holonomy

Reproduction of Examples 7.1 and 7.2 from the paper on finite AB arbitrage,
with verification of Propositions 1–4 and extended experiments.

All computations use **exact rational arithmetic** (`fractions.Fraction`),
ensuring bit-exact agreement with the paper's analytical results.

## Overview

The paper defines **AB arbitrage** through the *holonomy* of a loop in a time
category T = {t₀, t₁, t₂}. Each arrow iₖ: t_{k-1} → t_k induces a measurable
map F(iₖ): Ω_{tₖ} → Ω_{t_{k-1}} between probability spaces (in the opposite
direction). The **distortion operator** dF(iₖ) is the Radon-Nikodym derivative
of the pushforward measure. The **holonomy** Hol(γ) of the full loop is computed
recursively via the simplicial distortion formula:

```
Hol(γ) = dF(i₁) · CE(i₁, dF(i₂) · CE(i₂, dF(i₃)))
```

If P(Hol(γ) > 1) > 0, the loop exhibits **AB arbitrage**.

## Repository Structure

```
repo/
├── src/
│   ├── __init__.py
│   └── holonomy.py          # Core: ProbSpace, CE, distortion, simplicial_distortion
├── exp/
│   ├── __init__.py
│   ├── exp_1.py             # Example 7.1 — Hol = {0:0, 1:4}
│   ├── exp_2.py             # Example 7.2 — Hol = {0:1, 1:2}
│   ├── exp_3.py             # Proposition 1: dF(i∘j) = CE(i, dF(j))
│   ├── exp_4.py             # Propositions 2, 3, 4
│   ├── exp_5.py             # Parameter sensitivity analysis
│   └── exp_6.py             # Extended: 4-arrow, multi-loop, FX triangle
├── tests/
│   ├── test_holonomy.py     # Unit tests for src/holonomy.py
│   └── test_experiments.py  # Integration tests for all experiments
├── data/
│   └── .gitkeep             # No empirical data (all analytically specified)
├── results/
│   └── RESULTS.md           # Summary of all experiment findings
├── run_all_experiments.py   # Main runner
├── README.md
└── .gitignore
```

## Experiments

| # | Title | Expected Result |
|---|---|---|
| 1 | Example 7.1: Simple AB Arbitrage | Hol = {0:0, 1:4}, E[V^AB] = 5/2 |
| 2 | Example 7.2: Stronger AB Arbitrage | Hol = {0:1, 1:2}, E[V^AB] = 7/4 |
| 3 | Proposition 1: Multiplicativity | dF(i∘j) = CE(i, dF(j)) for all pairs |
| 4 | Propositions 2, 3, 4 | Trivial→trivial, V^AB≥1, V^wAB≥1 |
| 5 | Parameter Sensitivity | Hol invariant to probability weights |
| 6 | Extended Loops (4-arrow, multi-loop, FX) | 4-arrow: Hol={0:104/9, 1:0} |

## Usage

### Run All Experiments

```bash
python run_all_experiments.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Individual Experiment

```bash
python exp/exp_1.py
python exp/exp_2.py
# etc.
```

## Key Results

- **Example 7.1**: Hol(γ) = {0: 0, 1: 4} regardless of μ₂(0) ∈ (0,1). E[V^AB] = 5/2.
- **Example 7.2**: Hol(γ) = {0: 1, 1: 2}; min(Hol) ≥ 1 (stronger AB). E[V^AB] = 7/4.
- **Holonomy Invariance**: In both examples, holonomy values are structural invariants
  of the measurable maps, independent of probability weights.
- **4-Arrow Loop**: Hol(γ₄) = {0: 104/9, 1: 0} — longer loops accumulate larger holonomy.

## Dependencies

- Python 3.10+
- `fractions` (standard library) — exact rational arithmetic
- `pandas` — table display
- `matplotlib` — plots
- `pytest` — testing

No external quantitative libraries required (the methodology is custom).
