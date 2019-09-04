import dolfin as df
from ufl import grad as ufl_grad

import pulse

from perspect.porousproblem import PorousProblem


class Perspect(object):

    def __init__(self, geometry, parameters=None):
        mprob = PorousProblem(geometry, parameters=parameters,
                        solver_parameters={'newton_solver': {'linear_solver': 'mumps'}})
