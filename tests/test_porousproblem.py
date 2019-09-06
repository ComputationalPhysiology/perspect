import perspect
import dolfin as df
import pytest

def test_porousproblem(geometry):
    p = perspect.PorousProblem(geometry)
    assert 1==1


@pytest.fixture
def geometry():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])
    return geometry
