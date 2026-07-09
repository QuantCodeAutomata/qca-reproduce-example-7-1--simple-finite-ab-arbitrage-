"""
Core module for computing holonomy of loops in categorical probability spaces.

Implements the simplicial distortion operator and AB-arbitrage analysis
as described in the paper (Examples 7.1, 7.2, Propositions 1-4).

All computations use exact rational arithmetic via fractions.Fraction.

# Custom — Context7 found no library equivalent for categorical probability holonomy.
# This implements the distortion operator and simplicial holonomy from scratch.
"""

from fractions import Fraction
from typing import Any


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class ProbSpace:
    """
    Finite probability space (Ω, μ).

    Parameters
    ----------
    name : str
        Label for the space (e.g. 't0', 't1', 'USD').
    omega : list
        Elements of the sample space.
    mu : dict[Any, Fraction]
        Probability measure; mu[ω] must be Fraction and sum to 1.
    """

    def __init__(self, name: str, omega: list, mu: dict) -> None:
        self.name = name
        self.omega = omega
        self.mu = {k: Fraction(v) for k, v in mu.items()}
        total = sum(self.mu.values())
        assert total == Fraction(1, 1), (
            f"Probabilities in '{name}' sum to {total}, not 1."
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"ProbSpace({self.name}, mu={dict(self.mu)})"


# ---------------------------------------------------------------------------
# Core Operators
# ---------------------------------------------------------------------------

def conditional_expectation(
    arrow_name: str,
    f: dict,
    spaces: dict,
    arrow_info: dict,
) -> dict:
    """
    Compute the conditional expectation CE(i_k)(f).

    Given arrow i_k (T-arrow t_{k-1}→t_k, T^op-arrow t_k→t_{k-1}):
      F(i_k) : Ω_{source} → Ω_{target}  (source_space → target_space in arrow_info)

    For each y in Ω_{target}:
      CE[y] = Σ_{x : F(x)=y} f[x] · μ_{source}[x]  /  μ_{target}[y]

    Parameters
    ----------
    arrow_name : str
        Key in arrow_info (e.g. 'i1', 'i2', 'i3').
    f : dict
        Function defined on source_space.omega; values are Fraction.
    spaces : dict[str, ProbSpace]
        All probability spaces keyed by name.
    arrow_info : dict
        Metadata for each arrow:
        {arrow_name: {'F_map': dict, 'source_space': str, 'target_space': str}}
        source_space = Ω from which F(i_k) maps (T^op-source)
        target_space = Ω into which F(i_k) maps (T^op-target)

    Returns
    -------
    dict
        CE values keyed by elements of target_space.omega.

    Raises
    ------
    ValueError
        If target measure is zero at y but preimage is non-empty (inadmissible).
    """
    info = arrow_info[arrow_name]
    F_map = info["F_map"]
    source_sp: ProbSpace = spaces[info["source_space"]]
    target_sp: ProbSpace = spaces[info["target_space"]]

    result: dict = {}
    for y in target_sp.omega:
        preimage = [x for x in source_sp.omega if F_map[x] == y]
        numerator = sum(
            (Fraction(f[x]) * source_sp.mu[x] for x in preimage),
            Fraction(0),
        )
        mu_y = target_sp.mu[y]
        if mu_y > Fraction(0):
            result[y] = numerator / mu_y
        elif numerator > Fraction(0):
            raise ValueError(
                f"Inadmissible: μ_{target_sp.name}({y})=0 but preimage "
                f"{preimage} has positive mass."
            )
        else:
            # 0/0 convention: define as 0 (null set)
            result[y] = Fraction(0)
    return result


def distortion(arrow_name: str, spaces: dict, arrow_info: dict) -> dict:
    """
    Compute the distortion operator dF(i_k).

    dF(i_k)[y] = (F(i_k)_* μ_{source})[y] / μ_{target}[y]
               = CE(i_k)(1_{source})[y]

    where 1_{source} is the constant-1 function on source_space.omega.

    Parameters
    ----------
    arrow_name : str
        Key in arrow_info.
    spaces : dict[str, ProbSpace]
        All probability spaces.
    arrow_info : dict
        Arrow metadata.

    Returns
    -------
    dict
        Distortion values on target_space.omega.
    """
    info = arrow_info[arrow_name]
    source_sp: ProbSpace = spaces[info["source_space"]]
    ones_f = {x: Fraction(1) for x in source_sp.omega}
    return conditional_expectation(arrow_name, ones_f, spaces, arrow_info)


def pointwise_product(f1: dict, f2: dict) -> dict:
    """
    Compute the pointwise product of two L^∞ functions with the same domain.

    Parameters
    ----------
    f1, f2 : dict
        Functions with identical key sets.

    Returns
    -------
    dict
        {ω: f1[ω] * f2[ω]} for ω in shared domain.
    """
    assert set(f1.keys()) == set(f2.keys()), (
        f"Domain mismatch: {set(f1.keys())} vs {set(f2.keys())}"
    )
    return {omega: Fraction(f1[omega]) * Fraction(f2[omega]) for omega in f1}


def simplicial_distortion(
    sigma_objects: list,
    arrows_in_loop: list,
    spaces: dict,
    arrow_info: dict,
) -> dict:
    """
    Recursively compute the simplicial distortion dF(σ).

    For σ = [t_0, t_1, ..., t_n] with arrows [i_1, ..., i_n]:

      n=0: trivial distortion {ω: 1} on Ω_{t_0}
      n=1: dF(i_1) on Ω_{sigma_objects[0]} (= target_space of i_1)
      n>1: dF(i_1) · CE(i_1, dF(σ_tail))   where σ_tail = [t_1,...,t_n]

    The result always lives on spaces[sigma_objects[0]].omega via the
    target_space of the first arrow (which equals sigma_objects[0]).

    Parameters
    ----------
    sigma_objects : list of str
        Ordered list of space names (may repeat for loops, e.g. 't0' at start & end).
    arrows_in_loop : list of str
        Arrow names corresponding to consecutive pairs in sigma_objects.
        len(arrows_in_loop) == len(sigma_objects) - 1.
    spaces : dict[str, ProbSpace]
    arrow_info : dict

    Returns
    -------
    dict
        Simplicial distortion values on spaces[sigma_objects[0]].omega.
    """
    n = len(sigma_objects) - 1
    assert len(arrows_in_loop) == n, (
        f"Expected {n} arrows, got {len(arrows_in_loop)}."
    )

    if n == 0:
        sp = spaces[sigma_objects[0]]
        return {omega: Fraction(1) for omega in sp.omega}

    if n == 1:
        return distortion(arrows_in_loop[0], spaces, arrow_info)

    # n > 1: recursive case
    first_arrow = arrows_in_loop[0]
    tail_sigma = sigma_objects[1:]
    tail_arrows = arrows_in_loop[1:]

    tail_dist = simplicial_distortion(tail_sigma, tail_arrows, spaces, arrow_info)
    term1 = distortion(first_arrow, spaces, arrow_info)
    term2 = conditional_expectation(first_arrow, tail_dist, spaces, arrow_info)
    return pointwise_product(term1, term2)


# ---------------------------------------------------------------------------
# Composed Maps (for Proposition 1 verification)
# ---------------------------------------------------------------------------

def compose_maps(F_outer: dict, F_inner: dict, source_omega: list) -> dict:
    """
    Compose two measurable maps: F_outer ∘ F_inner.

    F_inner maps Ω_source → Ω_middle.
    F_outer maps Ω_middle → Ω_target.
    Returns {x: F_outer[F_inner[x]] for x in source_omega}.

    Parameters
    ----------
    F_outer : dict
        Map from middle space to target space.
    F_inner : dict
        Map from source space to middle space.
    source_omega : list
        Elements of the source space.

    Returns
    -------
    dict
        Composed map from source_omega to target space.
    """
    return {x: F_outer[F_inner[x]] for x in source_omega}


def distortion_of_composed_map(
    composed_F_map: dict,
    source_space: ProbSpace,
    target_space: ProbSpace,
) -> dict:
    """
    Compute dF(composed arrow) directly via the Radon-Nikodym formula.

    dF[y] = (pushforward μ_source)[y] / μ_target[y]

    Parameters
    ----------
    composed_F_map : dict
        {source_elem: target_elem} for the composed measurable map.
    source_space : ProbSpace
        Domain of the composed map.
    target_space : ProbSpace
        Codomain of the composed map.

    Returns
    -------
    dict
        Distortion values on target_space.omega.

    Raises
    ------
    ValueError
        If null-preservation is violated (μ_target[y]=0 but pushforward[y]>0).
    """
    pushforward: dict = {y: Fraction(0) for y in target_space.omega}
    for x in source_space.omega:
        y = composed_F_map[x]
        pushforward[y] += source_space.mu[x]

    result: dict = {}
    for y in target_space.omega:
        mu_y = target_space.mu[y]
        push_y = pushforward[y]
        if mu_y > Fraction(0):
            result[y] = push_y / mu_y
        elif push_y > Fraction(0):
            raise ValueError(
                f"Null-preservation violated: μ_target({y})=0 but pushforward={push_y}."
            )
        else:
            result[y] = Fraction(0)
    return result


# ---------------------------------------------------------------------------
# AB Arbitrage Strategy
# ---------------------------------------------------------------------------

def compute_ab_strategy(
    hol: dict,
    mu_base: dict,
) -> tuple[dict, dict, Fraction, Fraction]:
    """
    Construct the AB trading strategy θ^AB and compute terminal wealth V^AB.

    θ^AB(ω) = 1 if Hol(γ)(ω) > 1, else 0.
    V^AB(ω) = 1 + θ^AB(ω) · (Hol(γ)(ω) − 1).

    Parameters
    ----------
    hol : dict
        Holonomy values {ω: Fraction}.
    mu_base : dict
        Base measure on the same domain {ω: Fraction}.

    Returns
    -------
    theta_AB, V_AB, E_V_AB, E_profit : tuple
    """
    theta_AB = {
        omega: Fraction(1) if hol[omega] > Fraction(1) else Fraction(0)
        for omega in hol
    }
    V_AB = {
        omega: Fraction(1) + theta_AB[omega] * (hol[omega] - Fraction(1))
        for omega in hol
    }
    E_V_AB = sum(V_AB[omega] * Fraction(mu_base[omega]) for omega in hol)
    E_profit = E_V_AB - Fraction(1)
    return theta_AB, V_AB, E_V_AB, E_profit


def compute_weak_ab_strategy(
    hol: dict,
    mu_base: dict,
    epsilon: Fraction,
) -> tuple[dict, dict]:
    """
    Construct the weak AB trading strategy θ^wAB.

    θ^wAB(ω) = +1  if Hol(γ)(ω) > 1 + ε
               -1  if Hol(γ)(ω) < 1 - ε  (reverse execution, may be inadmissible)
                0  otherwise

    V^wAB(ω) = 1 + θ^wAB(ω) · (Hol(γ)(ω) − 1).

    Parameters
    ----------
    hol : dict
        Holonomy values.
    mu_base : dict
        Base probability measure.
    epsilon : Fraction
        Threshold parameter ε ≥ 0.

    Returns
    -------
    theta_wAB, V_wAB : tuple of dicts
    """
    theta_wAB: dict = {}
    for omega in hol:
        h = hol[omega]
        if h > Fraction(1) + epsilon:
            theta_wAB[omega] = Fraction(1)
        elif h < Fraction(1) - epsilon:
            theta_wAB[omega] = Fraction(-1)
        else:
            theta_wAB[omega] = Fraction(0)

    V_wAB = {
        omega: Fraction(1) + theta_wAB[omega] * (hol[omega] - Fraction(1))
        for omega in hol
    }
    return theta_wAB, V_wAB


# ---------------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------------

def check_normalization(
    df_values: dict,
    mu_target: dict,
) -> Fraction:
    """
    Verify sum_{y} dF(i_k)[y] · μ_target[y] == 1.

    Returns the sum (should equal Fraction(1,1)).
    """
    return sum(
        Fraction(df_values[y]) * Fraction(mu_target[y]) for y in df_values
    )


def check_ce_integral_preservation(
    arrow_name: str,
    f: dict,
    spaces: dict,
    arrow_info: dict,
) -> tuple[Fraction, Fraction]:
    """
    Verify CE integral preservation:
      Σ_y CE(i_k,f)[y] · μ_target[y]  ==  Σ_x f[x] · μ_source[x]

    Returns (lhs, rhs) as Fractions; they should be equal.
    """
    info = arrow_info[arrow_name]
    source_sp: ProbSpace = spaces[info["source_space"]]
    target_sp: ProbSpace = spaces[info["target_space"]]

    ce_vals = conditional_expectation(arrow_name, f, spaces, arrow_info)
    lhs = sum(
        Fraction(ce_vals[y]) * target_sp.mu[y] for y in target_sp.omega
    )
    rhs = sum(
        Fraction(f[x]) * source_sp.mu[x] for x in source_sp.omega
    )
    return lhs, rhs


def compute_ab_metrics(hol: dict, mu_base: dict) -> dict:
    """
    Compute AB arbitrage probability metrics from holonomy.

    Returns dict with keys:
      P_AB      : P(Hol > 1)
      P_no_loss : P(Hol >= 1)
      P_weak_AB : P(Hol != 1)
      E_Hol     : E[Hol]
    """
    mu = {k: Fraction(v) for k, v in mu_base.items()}
    one = Fraction(1)
    P_AB = sum(mu[w] for w in hol if hol[w] > one)
    P_no_loss = sum(mu[w] for w in hol if hol[w] >= one)
    P_weak_AB = sum(mu[w] for w in hol if hol[w] != one)
    E_Hol = sum(Fraction(hol[w]) * mu[w] for w in hol)
    return {
        "P_AB": P_AB,
        "P_no_loss": P_no_loss,
        "P_weak_AB": P_weak_AB,
        "E_Hol": E_Hol,
    }
