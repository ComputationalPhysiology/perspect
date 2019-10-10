import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, material, mechanics_bcs=None, porous_bcs=None,
                    parameters=None, solver_parameters=None):
        self.geometry = geometry
        self.material = material
        self.pprob = PorousProblem(geometry, material, bcs=porous_bcs,
                                    parameters=parameters,
                                    solver_parameters=solver_parameters)
        self.mprob = pulse.MechanicsProblem(geometry, material,
                                            bcs=mechanics_bcs,
                                            bcs_parameters={"": ""},
                                            solver_parameters=solver_parameters)

        self.pprob_state = self.pprob.state
        self.mprob_state = self.mprob.state
        self.mprob_state_prev = self.mprob.state

        # set pulse log level
        pulse.parameters.update({'log_level': df.get_log_level()})


    def update_mechanics(self):
        # calculate displacement velocity
        displacement, solid_pressure = self.mprob_state.split(deepcopy=True)
        pressure = self.calculate_pressure()
        mech_velocity = self.calculate_mech_velocity()
        self.pprob.update_mechanics(pressure, displacement, mech_velocity)
        self.mprob_state_prev.assign(self.mprob.state)


    def calculate_pressure(self):
        u, p = self.mprob_state.split()
        F = df.variable(pulse.kinematics.DeformationGradient(u))
        return df.project(df.inner(
                        df.diff(self.material.strain_energy(F), F), F.T) - p,
                    self.pprob.state_space)


    def calculate_mech_velocity(self):
        u, p = self.mprob_state.split()
        u_prev, p_prev = self.mprob_state_prev.split()
        dt = self.pprob.parameters['dt']/self.pprob.parameters['steps']
        return df.project((u-u_prev)/dt, self.pprob.vector_space)


    def solve(self):
        if self.pprob.parameters['mechanics']:
            self.solve_mechanics()
            self.update_mechanics()
        self.solve_porous()


    def iterate(self, control, target, **kwargs):
        self.previous_mechanics = self.mprob.state.split(deepcopy=True)
        pulse.iterate.iterate(self.mprob, control, target, **kwargs)
        self.mechanics = self.mprob.state.split(deepcopy=True)


    def solve_mechanics(self):
        self.previous_mechanics = self.mprob.state.split(deepcopy=True)
        self.mprob.solve()
        self.mechanics = self.mprob.state.split(deepcopy=True)


    def solve_porous(self):
        self.pprob.solve()
