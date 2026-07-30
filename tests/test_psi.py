"""Unit tests for the PSI (Population Stability Index) calculation."""

import numpy as np
import pytest

from fleetsense.monitoring.distribution_monitoring import psi


def test_psi_identical_distributions_is_zero():
    """Identical proportions should give PSI ~ 0 (no drift)."""
    ref = np.array([0.25, 0.25, 0.25, 0.25])
    comp = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(ref, comp) == pytest.approx(0.0, abs=1e-9)


def test_psi_known_shift():
    """PSI on a hand-computed shift matches the manual calculation."""
    ref = np.array([0.5, 0.5])
    comp = np.array([0.9, 0.1])
    expected = (0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5)
    assert psi(ref, comp) == pytest.approx(expected, rel=1e-6)
