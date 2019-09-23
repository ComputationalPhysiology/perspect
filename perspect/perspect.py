import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, material, mechanics_bcs=None, porous_bcs=None,
                    parameters=None):
        self.pprob = PorousProblem(geometry, material, bcs=porous_bcs, parameters=parameters)
        self.mprob = pulse.MechanicsProblem(geometry, material, bcs=mechanics_bcs,
                                            bcs_parameters={"": ""})


    def update_mechanics(self, displacement):
        self.pprob.update_mechanics(displacement)


    def solve(self):
        self.solve_mechanics()
        mu, mp = self.mprob.state.split(deepcopy=True)
        self.update_mechanics(mu)
        self.solve_porous()


    def solve_mechanics(self):
        self.mprob.solve()


    def solve_porous(self):
        self.pprob.solve()
