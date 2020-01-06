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

        if self.parameters['N'] > 1 and\
                            len(self.parameters['K']) != self.parameters['N']:
            N = self.parameters['N']
            K = self.parameters['K']
            msg = "Parameter N is {}, but K contains {} elements."\
                    "K has to contain exactly N elements.".format(N, len(K))
            raise ValueError(msg)

        # Create function spaces
        self._init_spaces()
        self._init_form()
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
        self.state = Function(self.state_space, name="m")
        self.state_previous = Function(self.state_space)
        self.state_test = TestFunction(self.state_space)
        self.displacement = Function(self.vector_space, name="du")
        self.mech_velocity = Function(self.vector_space)
        self.pressure = Function(self.state_space, name="p")
        self.darcy_flow = [Function(self.vector_space, name="w") 
                                                            for i in range(N)]


    def _init_form(self):
        m = self.state
        m_n = self.state_previous
        v = self.state_test
        u = self.displacement
        du = self.mech_velocity
        p = self.pressure

        N = self.parameters['N']

        # Get parameters
        rho = Constant(self.parameters['rho'])
        beta = [Constant(beta) for beta in self.parameters['beta']]
        if isinstance(self.parameters['K'], float):
            K = [self.parameters['K']]
        else:
            K = self.parameters['K']
        if self.geometry.f0 is not None:
            self.K = [self.permeability_tensor(k) for k in K]
        else:
            self.K = [Constant(k) for k in K]
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

        if N == 1:
            self._form = k*(m - m_n)*v*dx
        else:
            self._form = sum([k*(m[i] - m_n[i])*v[i]*dx for i in range(N)])

        # porous dynamics
        if self.parameters['mechanics']:
            A = [J*df.inv(F)*K*df.inv(F.T) for K in self.K]
        else:
            A = [Constant(1.0)*K for K in self.K]

        if N == 1:
            self._form += df.div(-rho*A[0]*df.grad(p))*v*dx
        else:
            self._form += sum([
                        df.dot(-rho*A[i]*df.grad(p.sub(i)))*v[i]*dx
                                                            for i in range(N)])

        # compartment coupling
        if N > 1:
            # forward
            self._form -= sum([-J*beta[i]*(p[i]-p[i+1])*v[i]*dx
                                                        for i in range(N-1)])
            # backward
            self._form -= sum([-J*beta[i-1]*(p[i]-p[i-1])*v[i]*dx
                                                        for i in range(1, N)])


        # add mechanics
        if self.parameters['mechanics']:
            if N == 1:
                self._form += -df.dot(df.grad(M), du)*v*dx
            else:
                self._form += sum([-df.dot(df.grad(M[i]), du)*v[i]*dx
                                                            for i in range(N)])

        # Add inflow/outflow terms
        if N == 1:
            self._form -= rho*qi*v*dx - rho*qo*v*dx
        else:
            self._form -= rho*qi*v[0]*dx - rho*qo*v[-1]*dx

        # self._form = m*v*dx + df.div(-A[0]*df.grad(p))*v*dx - qi*v*dx


    def inflow_rate(self, rate):
        if isinstance(rate, (int, float)):
            rate = Constant(rate)
        elif isinstance(rate, str):
            rate = Expression(rate, degree=1)
        elif isinstance(rate, df.function.expression.Expression):
            rate = rate
        return rate


    def permeability_tensor(self, K):
        FS = VectorFunctionSpace(self.geometry.mesh, 'P', 1)
        d = self.geometry.dim()
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
        factor = 1
        if d == 3:
            ftensor = df.as_matrix(
                        ((fibers[0], sheet[0], csheet[0]),
                        (fibers[1], sheet[1], csheet[1]),
                        (fibers[2], sheet[2], csheet[2])))
            ktensor = diag(df.as_vector([K, K/factor, K/factor]))
        else:
            ftensor = df.as_matrix(
                        ((fibers[0], sheet[0]),
                        (fibers[1], sheet[1])))
            ktensor = diag(df.as_vector([K, K/factor]))

        permeability = df.dot(df.dot(ftensor, ktensor), df.inv(ftensor))
        return permeability


    def update_mechanics(self, pressure, displacement, mech_velocity):
        N = self.parameters['N']
        phi = self.parameters['phi']
        if N == 1:
            self.pressure.assign(phi[0]*pressure)
        else:
            for i in range(N):
                self.pressure.sub(i).assign(phi[i]*pressure)
        self.displacement.assign(displacement)
        self.mech_velocity.assign(mech_velocity)


    def prescribed_pressure(self, pe):
        # Calculate endo to epi permeability gradient
        w = df.interpolate(pe, self.pressure.function_space())
        self.pressure.assign(w)


    def calculate_darcy_flow(self):
        rho = Constant(self.parameters['rho'])
        F = df.variable(kinematics.DeformationGradient(self.displacement))
        J = kinematics.Jacobian(F)
        dx = self.geometry.dx
        N = self.parameters['N']

        # Calculate endo to epi permeability gradient
        w = [TrialFunction(self.vector_space) for i in range(N)]
        v = [TestFunction(self.vector_space) for i in range(N)]
        a = [(1/rho)*df.dot(df.dot(F, w[i]), v[i])*dx for i in range(N)]
        # porous dynamics
        if self.parameters['mechanics']:
            A = [J*self.K[i]*df.inv(F.T) for i in range(N)]
        else:
            A = [self.K[i] for i in range(N)]
        if N == 1:
            L = [df.dot(-A[0]*df.grad(self.pressure), v[0])*dx]
        else:
            L = [df.dot(-A[i]*df.grad(self.pressure.sub(i)), v[i])*dx
                                                            for i in range(N)]

        w = [Function(self.vector_space) for i in range(N)]
        [df.solve(a[i] == L[i], w[i], []) for i in range(N)]
        for i in range(N):
            self.darcy_flow[i].assign(w[i])


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
