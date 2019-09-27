from perspect import (HeartGeometry, HolzapfelOgden, Perspect, mesh_paths)
import pulse
import dolfin as df
import pytest

def test_perspect(perspect, geometry, material, mechanics_bcs):
    assert perspect.pprob.geometry == geometry
    assert perspect.mprob.geometry == geometry
    assert perspect.mprob.material == material
    assert perspect.mprob.bcs == mechanics_bcs


def test_solve(perspect, pulse_problem):
    perspect.solve()
    pu, pp = perspect.mprob.state.split(deepcopy=True)
    pulse_problem.solve()
    mu, mp = pulse_problem.state.split(deepcopy=True)
    assert df.errornorm(pu, mu) < 1e-10
    assert df.errornorm(pp, mp) < 1e-10



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
def mechanics_bcs(geometry):
    # LV Pressure
    lvp = df.Constant(0.0)
    lv_marker = geometry.markers['ENDO'][0]
    lv_pressure = pulse.NeumannBC(traction=lvp,
                                  marker=lv_marker, name='lv')
    neumann_bc = [lv_pressure]

    # Add spring term at the base with stiffness 1.0 kPa/cm^2
    base_spring = 1.0
    robin_bc = [
        pulse.RobinBC(
            value=df.Constant(base_spring), marker=geometry.markers["BASE"][0]
        )
    ]


    # Fix the basal plane in the longitudinal direction
    # 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = df.DirichletBC(
            V.sub(0), df.Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
        )
        return bc


    dirichlet_bc = [fix_basal_plane]

    bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                                   neumann=neumann_bc,
                                   robin=robin_bc)
    return bcs

@pytest.fixture
def perspect(geometry, material, mechanics_bcs):
    perspect = Perspect(geometry, material, mechanics_bcs=mechanics_bcs,
                                                parameters={'mechanics': True})
    return perspect

@pytest.fixture
def pulse_problem(geometry, material, mechanics_bcs):
    pulse_problem = pulse.MechanicsProblem(geometry, material, mechanics_bcs)
    return pulse_problem
