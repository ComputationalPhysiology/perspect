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
        mu, mp = self.mprob.state.split()
        self.pprob.update_mechanics(mu)


    def solve(self):
        self.solve_mechanics()
        mu, mp = self.mprob.state.split(deepcopy=True)
        self.update_mechanics()
        self.solve_porous()


    def iterate(self, control, target, **kwargs):
        pulse.iterate.iterate(self.mprob, control, target, **kwargs)


    def solve_mechanics(self):
        self.mprob.solve()


    def solve_porous(self):
        self.pprob.solve()
