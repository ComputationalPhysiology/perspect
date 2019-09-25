import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, material, mechanics_bcs=None, porous_bcs=None,
                    parameters=None):
        self.geometry = geometry
        self.material = material
        self.pprob = PorousProblem(geometry, material, bcs=porous_bcs, parameters=parameters)
        self.mprob = pulse.MechanicsProblem(geometry, material, bcs=mechanics_bcs,
                                            bcs_parameters={"": ""})

        # set pulse log log level
        pulse.parameters.update({'log_level': df.get_log_level()})


    def update_mechanics(self):
        displacement = self.mechanics
        previous_displacement = self.previous_mechanics
        self.pprob.update_mechanics(displacement, previous_displacement)


    def solve(self):
        self.solve_mechanics()
        self.update_mechanics()
        self.solve_porous()


    def iterate(self, control, target, **kwargs):
        self.previous_mechanics = self.mprob.state.split(deepcopy=True)[0]
        pulse.iterate.iterate(self.mprob, control, target, **kwargs)
        self.mechanics = self.mprob.state.split(deepcopy=True)[0]


    def solve_mechanics(self):
        self.previous_mechanics = self.mprob.state.split(deepcopy=True)[0]
        self.mprob.solve()
        self.mechanics = self.mprob.state.split(deepcopy=True)[0]


    def solve_porous(self):
        self.pprob.solve()
