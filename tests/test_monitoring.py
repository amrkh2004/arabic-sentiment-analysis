"""
Unit tests for Monitoring and Statistical Drift Detection.
"""

import numpy as np

from scripts.monitor_drift import calculate_psi


def test_psi_identical_distributions():
    """
    Verifies that Population Stability Index (PSI) is zero for identical distributions.
    """
    p = np.array([0.5, 0.3, 0.2])
    psi = calculate_psi(p, p)
    assert np.isclose(psi, 0.0, atol=1e-5)


def test_psi_shifted_distributions():
    """
    Verifies that PSI is significantly positive when class distributions diverge.
    """
    baseline = np.array([0.7, 0.2, 0.1])
    shifted = np.array([0.1, 0.2, 0.7])
    psi = calculate_psi(baseline, shifted)
    assert psi > 0.25  # High drift threshold


def test_numerical_psi_continuous():
    """
    Verifies that continuous PSI accurately calculates drift for numerical features
    such as review text lengths and calibrated confidence scores.
    """
    from scripts.monitor_drift import calculate_numerical_psi

    # Identical numerical samples
    ref_lengths = np.random.normal(loc=50, scale=10, size=500)
    psi_ident = calculate_numerical_psi(ref_lengths, ref_lengths)
    assert np.isclose(psi_ident, 0.0, atol=1e-3)

    # Shifted distribution (e.g. much longer texts)
    shifted_lengths = np.random.normal(loc=120, scale=15, size=500)
    psi_drift = calculate_numerical_psi(ref_lengths, shifted_lengths)
    assert psi_drift > 0.25
