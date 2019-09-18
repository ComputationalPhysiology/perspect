import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, material, bcs=None, parameters=None):
        self.pprob = PorousProblem(geometry, parameters=parameters)
        self.mprob = pulse.MechanicsProblem(geometry, material, bcs=bcs)


    def update_mechanics(self, displacement):
        self.pprob.update_mechanics(displacement)


    def solve():
        self.solve_mechanics()
        mu, mp = self.mprob.state.split(deepcopy=True)
        self.update_mechanics(mu)
        self.solve_porous()


    def solve_mechanics(self):
        self.mprob.solve()


    def solve_porous():
        self.pprob.solve(self)
