from perspect import (PorousProblem, HeartGeometry, HolzapfelOgden, mesh_paths)
import pulse
import dolfin as df
import pytest

def test_porousproblem(geometry, material):
    parameters = {'K': 1}
    p = PorousProblem(geometry, material, parameters=parameters)

    assert geometry == p.geometry
    assert geometry.mesh == p.mesh

    for key in parameters.keys():
        assert key in p.parameters
        assert p.parameters[key] == parameters[key]


def test_init_spaces(porous_problem):
    p = porous_problem
    assert p.state.function_space() == p.state_space
    assert p.state_previous.function_space() == p.state_space
    assert p.state_test.function_space() == p.state_space


def test_init_porous_form(porous_problem):
    p = porous_problem
    assert sum(df.assemble(p._form)[:]) == 0


def test_update_mechanics(porous_problem, mechanics_problem):
    mechanics_problem.solve()
    mu, mp = mechanics_problem.state.split(deepcopy=True)
    porous_problem.update_mechanics(mu)
    assert(df.errornorm(porous_problem.displacement, mu) < 1e-10)


@pytest.fixture
def geometry():
    geometry = HeartGeometry.from_file(mesh_paths["simple_ellipsoid"])
    return geometry

@pytest.fixture
def material(geometry):
    activation = df.Function(df.FunctionSpace(geometry.mesh, "R", 0))
    activation.assign(df.Constant(0.0))
    matparams = HolzapfelOgden.default_parameters()
    material = HolzapfelOgden(activation=activation,
                                    parameters=matparams,
                                    active_model="active_stress",
                                    eta=0.3,
                                    f0=geometry.f0,
                                    s0=geometry.s0,
                                    n0=geometry.n0)
    return material

@pytest.fixture
def parameters():
    parameters = {'K': 1}
    return parameters

@pytest.fixture
def porous_problem(geometry, material, parameters):
    porous_problem = PorousProblem(geometry, material, parameters=parameters)
    return porous_problem

@pytest.fixture
def mechanics_problem(geometry, material):
    mechanics_problem = pulse.MechanicsProblem(geometry, material, bcs=None,
                                            bcs_parameters={"": ""})
    return mechanics_problem
