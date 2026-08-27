import numpy as np
import pytest

from photonpairlab.spdc.analysis import (
    quadratic_fit,
    quadratic_intersection_coordinates,
)


def test_quadratic_intersection_recovers_known_crossing_with_curvature():
    # Two curves with real curvature (not lines) that cross at x=30.
    x = np.linspace(10, 50, 25)
    y1 = 0.002 * (x - 30) ** 2 + 0.5 * (x - 30) + 1550.0  # increasing, curved
    y2 = -0.001 * (x - 30) ** 2 - 0.6 * (x - 30) + 1550.0  # decreasing, curved
    popt1, _ = quadratic_fit(x, y1)
    popt2, _ = quadratic_fit(x, y2)
    x_int, y_int = quadratic_intersection_coordinates(*popt1, *popt2, x_range=(x.min(), x.max()))
    assert x_int == pytest.approx(30.0, abs=1e-6)
    assert y_int == pytest.approx(1550.0, abs=1e-6)


def test_quadratic_intersection_picks_root_within_range():
    # a=1,b=0,c=-4 vs a=0,b=0,c=0 -> curves y=x^2-4 and y=0 cross at x=-2 and x=2.
    x_int, y_int = quadratic_intersection_coordinates(1, 0, -4, 0, 0, 0, x_range=(0, 10))
    assert x_int == pytest.approx(2.0)
    assert y_int == pytest.approx(0.0)


def test_quadratic_intersection_returns_nan_when_no_real_root():
    # y = x^2 + 1 vs y = 0 never intersect for real x.
    x_int, y_int = quadratic_intersection_coordinates(1, 0, 1, 0, 0, 0)
    assert np.isnan(x_int)
    assert np.isnan(y_int)
