import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, material, mechanics_bcs=None, porous_bcs=None,
                    parameters=None):
        self.geometry = geometry
        self.material = material
        self.pprob = PorousProblem(geometry, material, bcs=porous_bcs,
                                                        parameters=parameters)
        self.mprob = pulse.MechanicsProblem(geometry, material,
                                    bcs=mechanics_bcs, bcs_parameters={"": ""})

        # set pulse log level
        pulse.parameters.update({'log_level': df.get_log_level()})


    def update_mechanics(self):
        self.pprob.update_mechanics(self.mechanics, self.previous_mechanics)


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
