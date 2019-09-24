from collections import namedtuple

import dolfin as df
from dolfin import (
        Constant, FiniteElement, FunctionSpace, Function, TestFunction,
        TrialFunction, Identity, VectorElement, MixedElement,
        NonlinearVariationalProblem, NonlinearVariationalSolver
)

from pulse import HeartGeometry
from pulse import kinematics
from pulse.utils import get_lv_marker, set_default_none
import pulse.mechanicsproblem


logger = pulse.mechanicsproblem.logger


class PorousProblem(object):
    """
    Boundary marker labels:
    - inflow (Neumann BC in fluid mass increase)
    - outflow (Neumann BC in fluid mass increase)
    """

    def __init__(self, geometry, material, parameters=None,
                solver_parameters=None, **kwargs):
        self.geometry = geometry
        self.material = material
        self.mesh = geometry.mesh
        self.markers = get_lv_marker(self.geometry)

        # Set parameters
        self.parameters = PorousProblem.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Create function spaces
        self._init_spaces()
        self._init_porous_form()

        # Set up solver
        self.solver_parameters = PorousProblem.default_solver_parameters()
        if solver_parameters is not None:
            self.solver_parameters.update(solver_parameters)



    @staticmethod
    def default_parameters():
        """
        Default parameters for the porous problem.

        Taken from [Cookson2012, Michler2013]
        """

        return {
            'N': 1, 'rho': 1000, 'K': 1e-3, 'phi': 0.021, 'beta': 0.02,
            'qi': 0.0, 'qo': 0.0, 'tf': 1.0, 'dt': 1e-2, 'theta': 0.5,
            'mechanics': False
        }


    @staticmethod
    def default_solver_parameters():
        return NonlinearVariationalSolver.default_parameters()


    def _init_spaces(self):
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 2)
        N = self.parameters['N']
        if N == 1:
            elem = P2
        else:
            elem = MixedElement([P2 for i in range(N)])
        v_elem = VectorElement('P', self.mesh.ufl_cell(), 1)

        mesh = self.geometry.mesh
        self.state_space = FunctionSpace(mesh, elem)
        self.vector_space = FunctionSpace(mesh, v_elem)
        self.state = Function(self.state_space)
        self.state_previous = Function(self.state_space)
        self.state_test = TestFunction(self.state_space)
        self.displacement = Function(self.vector_space)
        self.pressure = Function(self.state_space)


    def _init_porous_form(self):
        m = self.state
        m_n = self.state_previous
        v = self.state_test
        du = self.displacement
        p = self.pressure

        # Multi-compartment functionality comes later
        if self.parameters['N'] > 1:
            m = self.state[0]
            m_n = self.state_previous[0]
            v = self.state_test[0]
            p = self.pressure[0]

        # Get parameters
        qi = self.parameters['qi']
        qo = self.parameters['qo']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        K = self.parameters['K']
        dt = self.parameters['dt']
        k = df.Constant(1/dt)
        theta = self.parameters['theta']

        # Crank-Nicolson time scheme
        M = theta*m + (1-theta)*m_n

        # Mechanics
        from ufl import grad as ufl_grad
        dx = self.geometry.dx
        d = self.state.geometric_dimension()
        I = Identity(d)
        F = df.variable(kinematics.DeformationGradient(du))
        J = kinematics.Jacobian(F)

        # porous dynamics
        if self.parameters['mechanics']:
            A = df.variable(rho * J * df.inv(F) * K * df.inv(F.T))
        else:
            A = rho*K

        self._form = k*(m - m_n)*v*dx +\
                            df.inner(-A*df.grad(p), df.grad(v))*dx

        # add mechanics
        if self.parameters['mechanics']:
            self._form += df.dot(df.grad(M), k*du)*v*dx

        # Add inflow/outflow terms
        self._form += -rho*qi*v*dx + rho*qo*v*dx


    def update_mechanics(self, displacement):
        self.displacement = displacement
        F = df.variable(kinematics.DeformationGradient(self.displacement))
        self.pressure = df.project(df.inner(df.diff(
                    self.material.strain_energy(F), F), F.T), self.state_space)
        self._init_porous_form()


    def solve(self):
        r"""
        Solve the variational problem

        """

        self._jacobian = df.derivative(
            self._form, self.state, TrialFunction(self.state_space)
        )

        logger.debug("Solving porous problem")
        # Get old state in case of non-convergence
        old_state = self.state.copy(deepcopy=True)

        problem = NonlinearVariationalProblem(
            self._form, self.state, J=self._jacobian
        )

        solver = NonlinearVariationalSolver(problem)
        solver.parameters.update(self.solver_parameters)

        try:
            logger.debug("Trying...")
            nliter, nlconv = solver.solve()
            if not nlconv:
                logger.debug("Failed")
                raise SolverDidNotConverge("Solver did not converge...")

        except RuntimeError as ex:
            logger.debug("Failed")
            logger.debug("Reintialize old state and raise exception")

            raise SolverDidNotConverge(ex)
        else:
            logger.debug("Solved")

            # Update old state
            self.state_previous.assign(self.state)

        return nliter, nlconv
