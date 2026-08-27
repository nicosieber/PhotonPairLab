import pytest

from photonpairlab.crystal.material.material_factory import MaterialFactory


@pytest.fixture
def bbo():
    return MaterialFactory.create("bbo")


def test_effective_index_reduces_to_ordinary_along_optic_axis(bbo):
    n_o = bbo.refractive_index(0.532, axis="o")
    assert bbo.effective_refractive_index(0.532, theta_deg=0) == pytest.approx(n_o)


def test_effective_index_reduces_to_extraordinary_perpendicular_to_optic_axis(bbo):
    n_e = bbo.refractive_index(0.532, axis="e")
    assert bbo.effective_refractive_index(0.532, theta_deg=90) == pytest.approx(n_e)


def test_effective_index_is_between_no_and_ne_at_intermediate_angles(bbo):
    n_o = bbo.refractive_index(0.532, axis="o")
    n_e = bbo.refractive_index(0.532, axis="e")
    n_45 = bbo.effective_refractive_index(0.532, theta_deg=45)
    lo, hi = sorted((n_o, n_e))
    assert lo < n_45 < hi
