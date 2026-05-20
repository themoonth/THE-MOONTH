"""
============================================================
SUPPLEMENTARY MATERIAL S1
============================================================
Title:   Monte Carlo Sensitivity Analysis for Alignment
         Probability Estimation
Paper:   Pentagonal Subdivision of a Proposed 29-Day
         Infradian Rhythm Produces Zero-Parameter Alignment
         with Documented Circasemiseptan and Circaseptan
         Cycles
Author:  Kamil Wojcik (Independent Researcher, Poland/Norway)
Email:   themoonthprotocol@gmail.com
Web:     themoonth.org

Pre-registration: https://osf.io/r3zet
OSF Project:      https://osf.io/jdzpc
GitHub:           https://github.com/themoonth

Reproduces: Table 2 of main manuscript
============================================================

DESCRIPTION
-----------
This script estimates the probability that a randomly chosen
subphase unit u — drawn from a specified prior distribution —
simultaneously aligns with both the circasemiseptan (~3.5 days)
and circaseptan (~7.0 days) documented infradian rhythms, within
the actual alignment precision of The Moonth framework (0.6%).

Five prior specifications are tested to demonstrate robustness
to prior choice. The subphase unit of The Moonth is derived
purely from the 29-day period and 5x5 nested structure:

    u = 29 / (5 x 5) = 29 / 25 = 1.16 days

No free parameters were adjusted to optimize alignment.

REQUIREMENTS
------------
Python >= 3.8
numpy  >= 1.20

USAGE
-----
    python Supplementary_Material_S1_MonteCarlo.py

EXPECTED RUNTIME
----------------
Approximately 60-120 seconds (10M trials x 5 priors).

EXPECTED OUTPUT
---------------
Reproduces Table 2 from main manuscript:
  Prior distribution          p      1 in N
  Uniform [0.5, 2.0] days    0.043    23
  Uniform [0.8, 1.5] days    0.035    29
  Uniform [1.0, 1.5] days    0.028    36
  Log-uniform [0.5, 2.0]     0.048    21
  Log-uniform [0.8, 1.5]     0.038    26

============================================================
"""

import numpy as np
import sys
import time

# ── Reproducibility ──────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Simulation parameters ─────────────────────────────────
N_TRIALS   = 10_000_000   # Monte Carlo trials per prior
TARGET_SEMI = 3.5          # Circasemiseptan period (days)
TARGET_SEP  = 7.0          # Circaseptan period (days)
TOLERANCE   = 0.006        # 0.6% — actual Moonth alignment precision

# ── The Moonth reference values ───────────────────────────
U_MOONTH   = 29.0 / 25.0  # = 1.16 days
N1_MOONTH  = round(TARGET_SEMI / U_MOONTH)   # = 3
N2_MOONTH  = round(TARGET_SEP  / U_MOONTH)   # = 6
DEV_SEMI_M = abs(N1_MOONTH * U_MOONTH - TARGET_SEMI) / TARGET_SEMI * 100
DEV_SEP_M  = abs(N2_MOONTH * U_MOONTH - TARGET_SEP)  / TARGET_SEP  * 100

# ── Prior specifications ──────────────────────────────────
# Each entry: (label, lower_bound, upper_bound, distribution_type)
PRIORS = [
    ("Uniform [0.5, 2.0] days",     0.5,  2.0,  "uniform"),
    ("Uniform [0.8, 1.5] days",     0.8,  1.5,  "uniform"),
    ("Uniform [1.0, 1.5] days",     1.0,  1.5,  "uniform"),
    ("Log-uniform [0.5, 2.0] days", 0.5,  2.0,  "log"),
    ("Log-uniform [0.8, 1.5] days", 0.8,  1.5,  "log"),
]


def sample_prior(lo, hi, dist, n):
    """Sample n values from specified prior distribution."""
    if dist == "uniform":
        return np.random.uniform(lo, hi, n)
    elif dist == "log":
        return np.exp(np.random.uniform(np.log(lo), np.log(hi), n))
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def run_monte_carlo(lo, hi, dist, n, target_semi, target_sep, tol):
    """
    Run Monte Carlo simulation for one prior specification.

    For each trial, draws a subphase unit u from the prior and
    checks whether integer multiples of u simultaneously align
    with both infradian targets within the specified tolerance.

    Returns probability of simultaneous alignment.
    """
    u  = sample_prior(lo, hi, dist, n)

    # Best integer multiples for each target
    n1 = np.round(target_semi / u).astype(int)
    n2 = np.round(target_sep  / u).astype(int)

    # Require positive multiples
    valid = (n1 >= 1) & (n2 >= 1)

    # Absolute deviations from targets
    tol_semi = tol * target_semi
    tol_sep  = tol * target_sep
    dev_semi = np.abs(n1 * u - target_semi)
    dev_sep  = np.abs(n2 * u - target_sep)

    # Simultaneous alignment within tolerance
    hits = valid & (dev_semi <= tol_semi) & (dev_sep <= tol_sep)

    return hits.sum() / n


# ── Header ────────────────────────────────────────────────
print("=" * 62)
print("SUPPLEMENTARY MATERIAL S1: Monte Carlo Sensitivity Analysis")
print("Wojcik (2026) — The Moonth Pentagonal Framework")
print("=" * 62)
print()
print(f"Python version:  {sys.version.split()[0]}")
print(f"NumPy version:   {np.__version__}")
print(f"Random seed:     {RANDOM_SEED}")
print(f"Trials per prior: {N_TRIALS:,}")
print()

# ── The Moonth reference ──────────────────────────────────
print("THE MOONTH SUBPHASE UNIT (zero-free-parameter derivation):")
print(f"  u = 29 / 25 = {U_MOONTH:.4f} days = {U_MOONTH*24:.2f} hours")
print()
print(f"  {N1_MOONTH}u = {N1_MOONTH * U_MOONTH:.4f} d  vs  circasemiseptan {TARGET_SEMI} d"
      f"  →  deviation: {DEV_SEMI_M:.3f}%")
print(f"  {N2_MOONTH}u = {N2_MOONTH * U_MOONTH:.4f} d  vs  circaseptan {TARGET_SEP} d"
      f"  →  deviation: {DEV_SEP_M:.3f}%")
print()
print(f"Tolerance applied: {TOLERANCE*100:.1f}% of each target")
print()

# ── Simulation ────────────────────────────────────────────
print("-" * 62)
print(f"{'Prior distribution':<35} {'p':>8}  {'1 in N':>7}")
print("-" * 62)

results = []
t0 = time.time()

for name, lo, hi, dist in PRIORS:
    p = run_monte_carlo(lo, hi, dist, N_TRIALS,
                        TARGET_SEMI, TARGET_SEP, TOLERANCE)
    n_in = round(1 / p) if p > 0 else float("inf")
    results.append((name, p, n_in))
    print(f"{name:<35} {p:>8.4f}  {n_in:>7.0f}")

elapsed = time.time() - t0
print("-" * 62)
print()

# ── Summary ───────────────────────────────────────────────
p_min = min(r[1] for r in results)
p_max = max(r[1] for r in results)
n_min = min(r[2] for r in results)
n_max = max(r[2] for r in results)

print("SUMMARY:")
print(f"  Probability range:  {p_min:.3f} – {p_max:.3f}")
print(f"  '1 in N' range:     1 in {n_max:.0f}  to  1 in {n_min:.0f}")
print(f"  Runtime:            {elapsed:.1f} seconds")
print()
print("INTERPRETATION:")
print("  The alignment probability is robust to prior choice.")
print("  Conservative priors (narrower range near u=1.16) produce")
print("  LOWER probabilities, confirming the result does not depend")
print("  on a permissive prior selection.")
print()
print("  Significance derives from three compounding factors:")
print("  (1) Zero-free-parameter derivation from 29-day period")
print("  (2) Both aligned rhythms are among the most consistently")
print("      documented infradian periodicities in chronobiology")
print("  (3) u = 27.84 hours falls within the Haus-Halberg")
print("      circadian range of 20-28 hours")
print()
print("=" * 62)
print("Reproduces Table 2 of main manuscript.")
print("Pre-registration: https://osf.io/r3zet")
print("=" * 62)
