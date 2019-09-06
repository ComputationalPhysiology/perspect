import perspect
import dolfin as df
import pytest

def test_porousproblem(geometry):
    parameters = {'K': 1}
    p = perspect.PorousProblem(geometry, parameters=parameters)

    assert geometry == p.geometry
    assert geometry.mesh == p.mesh

    for key in parameters.keys():
        assert key in p.parameters
        assert p.parameters[key] == parameters[key]


@pytest.fixture
def geometry():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])
    return geometry
