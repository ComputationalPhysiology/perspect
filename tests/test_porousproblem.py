from perspect import (PorousProblem, HeartGeometry, HolzapfelOgden, mesh_paths)
import pulse
import dolfin as df
import pytest

def test_porousproblem(geometry, material):
    parameters = {'K': 1}
    solver_parameters = {'newton_solver': {'maximum_iterations': 100}}
    p = PorousProblem(geometry, material, parameters=parameters,
                        solver_parameters=solver_parameters)

    assert geometry == p.geometry
    assert geometry.mesh == p.mesh

    assert 'K' in p.parameters
    assert p.parameters['K'] == parameters['K']

    assert p.solver_parameters['newton_solver']['maximum_iterations'] ==\
                    solver_parameters['newton_solver']['maximum_iterations']


def test_init_spaces(porous_problem, porous_problem2):
    p = porous_problem
    assert p.state.function_space() == p.state_space
    assert p.state_previous.function_space() == p.state_space
    assert p.state_test.function_space() == p.state_space
    p2 = porous_problem2
    assert p2.state.function_space() == p2.state_space
    assert p2.state_previous.function_space() == p2.state_space
    assert p2.state_test.function_space() == p2.state_space


def test_init_porous_form(porous_problem):
    p = porous_problem
    # assert sum(df.assemble(p._form)[:]) == 0


def test_update_mechanics(porous_problem, mechanics_problem):
    mu_prev, p_prev = mechanics_problem.state.split(deepcopy=True)
    mechanics_problem.solve()
    mu, p = mechanics_problem.state.split(deepcopy=True)
    dt = porous_problem.parameters['dt']
    vel = df.project((mu-mu_prev)/dt, porous_problem.vector_space)
    porous_problem.update_mechanics(mu, p, vel)
    assert(df.assemble(df.div(porous_problem.mech_velocity)*df.dx) < 1e-10)


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
def porous_problem2(geometry, material, parameters):
    parameters.update({'N': 2})
    porous_problem = PorousProblem(geometry, material, parameters=parameters)
    return porous_problem

@pytest.fixture
def mechanics_problem(geometry, material):
    mechanics_problem = pulse.MechanicsProblem(geometry, material, bcs=None,
                                            bcs_parameters={"": ""})
    return mechanics_problem
