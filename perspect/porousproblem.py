from collections import namedtuple

import numpy as np

import dolfin as df
from dolfin import (
        Constant, Expression, FiniteElement, FunctionSpace, DirichletBC,
        Function, TestFunction, TrialFunction, Identity, VectorElement,
        VectorFunctionSpace, MixedElement, NonlinearVariationalProblem,
        NonlinearVariationalSolver
)

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
        self.markers = self.geometry.markers
        self.isbiv = False
        if 'ENDO_RV' in self.markers.keys():
            self.isbiv = True

        # Set parameters
        self.parameters = PorousProblem.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Create function spaces
        self._init_spaces()
        self._init_porous_form()
        self.t = 0.0

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
            'N': 1, 'rho': 1.06, 'K': [1e-2], 'phi': [0.021], 'beta': [0.02],
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
        self.displacement = Function(self.vector_space)
        self.mech_velocity = Function(self.vector_space)
        self.pressure = Function(self.state_space)
        self.darcy_flow = Function(self.vector_space)


    def _init_porous_form(self):
        m = self.state
        m_n = self.state_previous
        v = self.state_test
        u = self.displacement
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
        beta = [Constant(beta) for beta in self.parameters['beta']]
        if self.geometry.f0 is not None:
            self.K = self.permeability_tensor()
        else:
            self.K = [Constant(K) for K in self.parameters['K']]
        dt = self.parameters['dt']/self.parameters['steps']
        qi = self.inflow_rate(self.parameters['qi'])
        qo = self.inflow_rate(self.parameters['qo'])
        k = Constant(1/dt)
        theta = self.parameters['theta']

        # Crank-Nicolson time scheme
        M = Constant(theta)*m + Constant(1-theta)*m_n

        # Mechanics
        from ufl import grad as ufl_grad
        dx = self.geometry.dx
        d = self.state.geometric_dimension()
        I = Identity(d)
        F = df.variable(kinematics.DeformationGradient(u))
        J = kinematics.Jacobian(F)

        self._form = k*(m - m_n)*v*dx

        # porous dynamics
        if self.parameters['mechanics']:
            A = J*df.inv(F)*self.K*df.inv(F.T)
        else:
            A = Constant(1.0)

        self._form += -df.inner(rho*A*df.grad(p), df.grad(v))*dx

        # add mechanics
        if self.parameters['mechanics']:
            self._form -= df.dot(df.grad(M), du)*v*dx

        # Add inflow/outflow terms
        self._form += -rho*qi*v*dx + rho*qo*v*dx


    def inflow_rate(self, rate):
        if isinstance(rate, (int, float)):
            rate = Constant(rate/self.mesh.num_cells())
        elif isinstance(rate, str):
            rate = Expression(rate, degree=1)/Constant(self.mesh.num_cells())
        return rate


    def permeability_tensor(self):
        FS = VectorFunctionSpace(self.geometry.mesh, 'P', 1)
        d = self.geometry.dim()
        K = self.parameters['K'][0]
        fibers = df.project((1/df.norm(self.geometry.f0)) * self.geometry.f0, FS)
        if self.geometry.s0 is not None:
            # normalize vectors
            sheet = self.geometry.s0 / df.norm(self.geometry.s0)
            if d == 3:
                csheet = self.geometry.n0 / df.norm(self.geometry.n0)
        else:
            # need to create two orthogonal vectors
            if d == 2:
                s0 = df.interpolate(Constant((1,1)), FS)
            elif d == 3:
                s0 = df.interpolate(Constant((1,1,1)), FS)
            sheet = df.project(s0 - df.dot(s0, fibers) * fibers, FS)
            sheet.vector().get_local()[:] /=\
                                np.linalg.norm(sheet.vector().get_local())
            csheet = df.cross(fibers, sheet)

        from ufl import diag
        if d == 3:
            ftensor = df.as_matrix(
                        ((fibers[0], sheet[0], csheet[0]),
                        (fibers[1], sheet[1], csheet[1]),
                        (fibers[2], sheet[2], csheet[2])))
            ktensor = diag(df.as_vector([K, K/10, K/10]))
        else:
            ftensor = df.as_matrix(
                        ((fibers[0], sheet[0]),
                        (fibers[1], sheet[1])))
            ktensor = diag(df.as_vector([K, K/10]))

        permeability = df.dot(df.dot(ftensor, ktensor), df.inv(ftensor))
        return permeability


    def update_mechanics(self, pressure, displacement, mech_velocity):
        self.pressure.assign(pressure)
        # self.test_pressure()
        self.displacement.assign(displacement)
        self.mech_velocity.assign(mech_velocity)


    def test_pressure(self):
        # Calculate endo to epi permeability gradient
        pe = Expression("1-x[0]*x[0]", degree=0)
        w = df.interpolate(pe, self.pressure.function_space())
        self.pressure.assign(w)


    def calculate_darcy_flow(self):
        rho = Constant(self.parameters['rho'])
        F = df.variable(kinematics.DeformationGradient(self.displacement))
        J = kinematics.Jacobian(F)
        dx = self.geometry.dx

        # Calculate endo to epi permeability gradient
        FS = VectorFunctionSpace(self.geometry.mesh, 'P', 1)
        w = TrialFunction(FS)
        v = TestFunction(FS)
        a = df.dot(F*w/rho, v)*dx
        # porous dynamics
        if self.parameters['mechanics']:
            L = df.dot(-J*self.K*df.inv(F.T)*df.grad(self.pressure), v) * dx
        else:
            L = df.dot(-self.K*df.grad(self.pressure), v) * dx

        w = Function(FS)
        df.solve(a == L, w, [])
        self.darcy_flow.assign(w)


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
                self.parameters['qi'].t +=\
                                self.parameters['dt']/self.parameters['steps']
            except AttributeError:
                # If the Expression for qi does not have a time parameter
                # do nothing
                pass

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

            self.newton_steps = nliter

        self.state_previous.assign(self.state)
        self.calculate_darcy_flow()

        return nliter, nlconv
