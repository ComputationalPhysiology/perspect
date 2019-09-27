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
        self.newton_steps = 5000
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
            'N': 1, 'rho': 1.06, 'K': 1, 'phi': 0.021, 'beta': 0.02e4,
            'qi': 0.0, 'qo': 0, 'tf': 1.0, 'dt': 1e-2, 'steps': 10,
            'theta': 0.5, 'mechanics': False
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

        mesh = self.mesh
        self.state_space = FunctionSpace(mesh, elem)
        self.vector_space = FunctionSpace(mesh, v_elem)
        self.state = Function(self.state_space)
        self.state_previous = Function(self.state_space)
        self.state_test = TestFunction(self.state_space)
        self.mech_velocity = Function(self.vector_space)
        self.pressure = Function(self.state_space)


    def _init_porous_form(self):
        m = self.state
        m_n = self.state_previous
        v = self.state_test
        du = self.mech_velocity
        p = self.pressure

        # Multi-compartment functionality comes later
        if self.parameters['N'] > 1:
            m = self.state[0]
            m_n = self.state_previous[0]
            v = self.state_test[0]
            p = self.pressure[0]

        # Get parameters
        rho = Constant(self.parameters['rho'])
        beta = Constant(self.parameters['beta'])
        K = Constant(self.parameters['K'])
        dt = self.parameters['dt']/self.parameters['steps']
        qi = Constant(self.parameters['qi']/self.mesh.num_cells()*dt)
        qo = Constant(self.parameters['qo']/self.mesh.num_cells()*dt)
        k = Constant(1/dt)
        theta = self.parameters['theta']

        # Crank-Nicolson time scheme
        M = Constant(theta)*m + Constant(1-theta)*m_n

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


    def update_mechanics(self, mechanics, previous_mechanics):
        displacement = mechanics[0]
        previous_displacement = previous_mechanics[0]
        self.mech_velocity.assign(
            df.project(displacement - previous_displacement, self.vector_space))
        F = df.variable(kinematics.DeformationGradient(displacement))
        mech_pressure = df.project(mechanics[1], self.state_space)
        self.pressure.assign(df.project(df.inner(df.diff(
                    self.material.strain_energy(F), F), F.T) - mech_pressure,
                                                            self.state_space))


    def solve(self):
        r"""
        Solve the variational problem

        """

        # Only recalculate Jacobian if Newton solver takes too many iterations
        if self.newton_steps > 10:
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

        for i in range(self.parameters['steps']):
            try:
                logger.debug("Trying...")
                nliter, nlconv = solver.solve()
                if not nlconv:
                    logger.debug("Failed")
                    logger.debug("Solver did not converge...")
                    break
            except RuntimeError as ex:
                nliter = 0
                nlconv = False
                logger.debug("Failed")
                logger.debug("Solver did not converge...")
                break

            else:
                logger.debug("Solved")

                # Update old state
                self.state_previous.assign(self.state)

            self.newton_steps = nliter

        return nliter, nlconv
