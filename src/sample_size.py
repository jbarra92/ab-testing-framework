"""
sample_size.py
--------------
Functions for statistical power analysis and sample size calculation
in the context of A/B tests on:
1) minimum_sample_size: binary outcomes (e.g. conversion rate).
2) minimum_sample_size_continuous: continuous outcomes (e.g. revenue per user)
"""

import math
import numpy as np
from scipy import stats


def minimum_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate the minimum sample size required per group for a two-sample
    proportion z-test.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate in the control group (between 0 and 1).
    mde : float
        Minimum detectable effect — the smallest absolute change in conversion
        rate worth detecting (e.g. 0.02 means detecting a lift from 5% to 7%).
    alpha : float, optional
        Significance level (Type I error rate). Default is 0.05.
    power : float, optional
        Statistical power (1 - Type II error rate). Default is 0.80.

    Returns
    -------
    int
        Minimum number of observations required in each group (control and
        treatment), rounded up to the nearest whole number.

    Notes
    -----
    Uses the normal approximation for two proportions::

        n = ( z_α/2 * √(2 * p̄(1−p̄))  +  z_β * √(p₁(1−p₁) + p₂(1−p₂)) )²
            ─────────────────────────────────────────────────────────────────
                                       (p₂ − p₁)²

    Examples
    --------
    >>> minimum_sample_size(baseline_rate=0.05, mde=0.02)
    2213
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2) # two-tailed z value
    z_beta = stats.norm.ppf(power) # z-value for desired power, e.g. 0.84 for 80% power
    p_hat = (baseline_rate + (baseline_rate + mde)) / 2 # pooled proportion
    p2 = baseline_rate + mde

    n = ((z_alpha * math.sqrt(2 * p_hat*(1-p_hat)) + z_beta * math.sqrt(baseline_rate*(1-baseline_rate) + p2*(1-p2)))**2)/(mde**2)

    return math.ceil(n) # return as int


def minimum_sample_size_continuous(
    std_dev: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate the minimum sample size required per group for a t-test for means.

    Parameters
    ----------
    std_dev : float
        Standard deviation .
    mde : float
        Minimum detectable effect — the smallest absolute change in conversion
        rate worth detecting (e.g. 0.02 means detecting a lift from 5% to 7%).
    alpha : float, optional
        Significance level (Type I error rate). Default is 0.05.
    power : float, optional
        Statistical power (1 - Type II error rate). Default is 0.80.

    Returns
    -------
    int
        Minimum number of observations required in each group (control and
        treatment), rounded up to the nearest whole number.

    Notes
    -----
    Uses the normal approximation for two proportions::

    n = (z_α/2 + z_β)² * 2σ²
        ─────────────────────
                δ²
    Examples
    --------
    >>> minimum_sample_size_continuous(std_dev=10.0, mde=1.0)
    1570
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2) # two-tailed z value
    z_beta = stats.norm.ppf(power) # z-value for desired power, e.g. 0.84 for 80% power
    var =  std_dev**2

    n = ((((z_alpha + z_beta)**2) * (2*var)))/(mde**2)

    return math.ceil(n) # return as int