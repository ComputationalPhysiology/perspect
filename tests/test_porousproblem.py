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


def test_init_spaces(porous_problem):
    p = porous_problem
    assert p.state.function_space() == p.state_space
    assert p.state_previous.function_space() == p.state_space
    assert p.state_test.function_space() == p.state_space


@pytest.fixture
def geometry():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])
    return geometry

@pytest.fixture
def bcs():
    bcs = None
    return bcs

@pytest.fixture
def parameters():
    parameters = {'K': 1}
    return parameters

@pytest.fixture
def porous_problem(geometry, bcs, parameters):
    porous_problem = perspect.PorousProblem(geometry, bcs=bcs, parameters=parameters)
    return porous_problem
