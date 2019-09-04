import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, **kwargs):
        mprob = PorousProblem(geometry,
                        solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})
