import perspect
import pulse
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


def test_init_porous_form(porous_problem):
    p = porous_problem
    assert sum(df.assemble(p._form)[:]) == 0


def test_update_mechanics(porous_problem, mechanics_problem):
    mechanics_problem.solve()
    mu, mp = mechanics_problem.state.split(deepcopy=True)
    assert(df.errornorm(porous_problem.displacement, mu) < 1e-5)


@pytest.fixture
def geometry():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])
    return geometry

@pytest.fixture
def material(geometry):
    activation = df.Function(df.FunctionSpace(geometry.mesh, "R", 0))
    activation.assign(df.Constant(0.0))
    matparams = pulse.HolzapfelOgden.default_parameters()
    material = pulse.HolzapfelOgden(activation=activation,
                                    parameters=matparams,
                                    active_model="active_stress",
                                    eta=0.3,
                                    f0=geometry.f0,
                                    s0=geometry.s0,
                                    n0=geometry.n0)
    return material

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

@pytest.fixture
def mechanics_problem(geometry, material, bcs):
    mechanics_problem = pulse.MechanicsProblem(geometry, material, bcs=bcs,
                                            bcs_parameters={"": ""})
    return mechanics_problem
