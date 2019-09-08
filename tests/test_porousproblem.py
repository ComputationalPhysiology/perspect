import perspect
import dolfin as df
import pytest

def test_porousproblem(geometry):
    parameters = {'K': 1}
    bcs = None
    p = perspect.PorousProblem(geometry, bcs=bcs, parameters=parameters)

    assert geometry == p.geometry
    assert geometry.mesh == p.mesh

    for key in parameters.keys():
        assert key in p.parameters
        assert p.parameters[key] == parameters[key]

    if bcs is None:
        from perspect.porousproblem import perfusion_boundary_conditions
        bcs = perfusion_boundary_conditions(geometry)
    for nbcs1, nbcs2 in zip(bcs.neumann, p.bcs.neumann):
        assert float(nbcs1.inflow) == float(nbcs2.inflow)
        assert nbcs1.marker == nbcs2.marker


@pytest.fixture
def geometry():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])
    return geometry
