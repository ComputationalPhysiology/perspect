import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


def get_mechanics_geometry(geometry):
    return pulse.geometry.HeartGeometry(geometry.mesh, markers=geometry.markers,
                                    marker_functions=geometry.markerfunctions,
                                    microstructure=geometry.microstructure,
                                    crl_basis=geometry.crl_basis)


class Perspect(object):

    def __init__(self, geometry, material, mechanics_bcs=None, porous_bcs=None,
                    parameters=None, solver_parameters=None):
        self.geometry = geometry
        self.material = material
        self.pprob = PorousProblem(geometry, material, bcs=porous_bcs,
                                    parameters=parameters,
                                    solver_parameters=solver_parameters)
        self.mprob = pulse.MechanicsProblem(get_mechanics_geometry(geometry),
                                            material, bcs=mechanics_bcs,
                                            bcs_parameters={"": ""})
        if solver_parameters is not None:
            self.mprob.solver_parameters.update(solver_parameters)

        # set pulse log level
        pulse.parameters.update({'log_level': df.get_log_level()})


    def update_mechanics(self):
        # update mechanics in porous problem
        displacement, solid_pressure = self.mprob.state.split(deepcopy=True)
        mech_velocity = self.calculate_mech_velocity(displacement)
        pressure = self.calculate_pressure(displacement, solid_pressure)
        self.pprob.update_mechanics(pressure, displacement, mech_velocity)


    def SecondPiolaStress(self, F, p=None, deviatoric=False):
        import dolfin
        from pulse import kinematics
        material = self.material
        I = kinematics.SecondOrderIdentity(F)

        f0 = material.f0
        f0f0 = dolfin.outer(f0, f0)

        I1 = dolfin.variable(material.active.I1(F))
        I4f = dolfin.variable(material.active.I4(F))

        Fe = material.active.Fe(F)
        Fa = material.active.Fa
        Ce = Fe.T * Fe

        # fe = Fe*f0
        # fefe = dolfin.outer(fe, fe)

        # Elastic volume ratio
        J = dolfin.variable(dolfin.det(Fe))
        # Active volume ration
        Ja = dolfin.det(Fa)

        dim = self.geometry.dim()
        Ce_bar = pow(J, -2.0 / float(dim)) * Ce

        w1 = material.W_1(I1, diff=1, dim=dim)
        w4f = material.W_4(I4f, diff=1)

        # Total Stress
        S_bar = Ja * (2 * w1 * I + 2 * w4f * f0f0) * dolfin.inv(Fa).T

        if material.is_isochoric:

            # Deviatoric
            Dev_S_bar = S_bar - (1.0 / 3.0) * dolfin.inner(S_bar, Ce_bar) * dolfin.inv(
                Ce_bar
            )

            S_mat = J ** (-2.0 / 3.0) * Dev_S_bar
        else:
            S_mat = S_bar

        # Volumetric
        if p is None or deviatoric:
            S_vol = dolfin.zero((dim, dim))
        else:
            psi_vol = material.compressibility(p, J)
            S_vol = J * dolfin.diff(psi_vol, J) * dolfin.inv(Ce)

        # Active stress
        wactive = material.active.Wactive(F, diff=1)
        eta = material.active.eta

        S_active = wactive * (f0f0 + eta * (I - f0f0))

        S = S_mat + S_vol + S_active

        return S


    def calculate_pressure(self, displacement, solid_pressure):
        if self.pprob.parameters['N'] == 1:
            pspace = self.pprob.state_space
        else:
            pspace = self.pprob.state_space.sub(0).collapse()
        F = df.variable(pulse.kinematics.DeformationGradient(displacement))
        return df.project(
                    -self.SecondPiolaStress(F, p=solid_pressure)[0, 0], pspace)


    def calculate_mech_velocity(self, displacement):
        dt = self.pprob.parameters['dt']/self.pprob.parameters['steps']
        return df.project(displacement/dt, self.pprob.vector_space)


    def iterate_mechanics(self, control, target, **kwargs):
        pulse.iterate.iterate(self.mprob, control, target, **kwargs)


    def solve_porous(self):
        self.pprob.solve()
