from collections import namedtuple

import dolfin as df
from dolfin import (
        Constant, FiniteElement, FunctionSpace, Function, TestFunction,
        Identity,
        NonlinearVariationalProblem, NonlinearVariationalSolver
)

from pulse import HeartGeometry
from pulse.utils import get_lv_marker

NeumannBC = namedtuple("NeumannBC", ["inflow", "marker", "name"])


def perfusion_boundary_conditions(geometry, inflow=0.0):
    # Neumann BC
    lv_marker = get_lv_marker(geometry)
    lv_inflow = NeumannBC(
        inflow=df.Constant(0.0, name="lv_inflow"), marker=lv_marker, name="lv"
    )
    neumann_bc = [lv_inflow]

    if geometry.is_biv:
        rv_pressure = NeumannBC(
            inflow=Constant(0.0, name="rv_inflow"),
            marker=geometry.markers["ENDO_RV"][0],
            name="rv",
        )

        neumann_bc += [rv_inflow]


class PorousProblem(object):
    """
    Boundary marker labels:
    - inflow (Neumann BC in fluid mass increase)
    - outflow (Neumann BC in fluid mass increase)
    """

    def __init__(self, geometry, bcs=None, parameters=None,
                solver_parameters=None, **kwargs):
        self.geometry = geometry
        self.mesh = geometry.mesh
        self.markers = get_lv_marker(self.geometry)

        # Set parameters
        self.parameters = PorousProblem.default_parameters()
        if parameters is not None:
            self.parameters.update(parameters)

        # Set boundary conditions
        if bcs is None:
            if isinstance(geometry, HeartGeometry):
                self.bcs_parameters = PorousProblem.default_bcs_parameters()
            self.bcs = perfusion_boundary_conditions(geometry,
                                                        **self.bcs_parameters)

        # Create function spaces
        self._init_spaces()
        self._init_porous_form()

        # Set up solver
        self.solver_parameters = PorousProblem.defaul_solver_parameters()
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
            'qi': 0.0, 'qo': 0.0, 'tf': 1.0, 'dt': 1e-2, 'theta': 0.5
        }


    @staticmethod
    def default_bcs_parameters():
        return dict(inflow=0.0)


    @staticmethod
    def defaul_solver_parameters():
        return NonlinearVariationalSolver.default_parameters()


    def _init_spaces(self):
        P2 = FiniteElement('P', self.mesh.ufl_cell(), 2)
        N = self.parameters['N']
        if N == 1:
            elem = P2
        else:
            elem = MixedElement([P1 for i in range(N)])

        mesh = self.geometry.mesh
        self.state_space = FunctionSpace(mesh, elem)
        self.state = Function(self.state_space)
        self.state_previous = Function(self.state_space)
        self.state_test = TestFunction(self.state_space)


    def _init_porous_form(self):
        m = self.state
        m_n = self.state_previous
        v = self.state_test
        p = Function(self.state_space)

        # Get parameters
        qi = self.parameters['qi']
        qo = self.parameters['qo']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        K = self.parameters['K']
        dt = self.parameters['dt']
        k = 1/dt
        theta = self.parameters['theta']
        theta_ = 1-theta

        # Crank-Nicolson time scheme
        M = theta*m + theta_*m_n

        # Mechanics
        dx = self.geometry.dx
        d = self.state.geometric_dimension()
        J = Constant(1)
        F = Identity(d)

        # porous dynamics
        A = df.variable(rho * J * df.inv(F) * K * df.inv(F.T))
        self._form = k*(m - m_n)*v*dx + df.inner(-A*df.grad(p), df.grad(v))*dx

        # mechanics
        # Form += dot(grad(M), k*(dU-dU_n))*v*dx

        # Add inflow/outflow terms
        self._form += -rho*qi*v*dx + rho*qo*v*dx


    def fluid_solid_coupling(self):
        TOL = self.TOL()
        dU, L = self.Us.split(True)
        if self.N == 1:
            FS = self.FS_M
        else:
            FS = self.FS_M.sub(0).collapse()
        for i in range(self.N):
            p = TrialFunction(self.FS_F)
            q = TestFunction(self.FS_F)
            a = p*q*dx
            Ll = (tr(diff(self.Psi, self.F) * self.F.T))/self.phif[i]*q*dx - L*q*dx
            A = assemble(a)
            b = assemble(Ll)
            [bc.apply(A, b) for bc in self.pbcs]
            solver = KrylovSolver('minres', 'hypre_amg')
            prm = solver.parameters
            prm.absolute_tolerance = TOL
            prm.relative_tolerance = TOL*1e3
            prm.maximum_iterations = 1000
            p = Function(self.FS_F)
            solver.solve(A, p.vector(), b)
            self.p[i].assign(project(p, FS))


    def calculate_flow_vector(self):
        FS = VectorFunctionSpace(self.mesh, 'P', 1)
        dU, L = self.Us.split(True)
        m = TrialFunction(self.FS_V)
        mv = TestFunction(self.FS_V)

        # Parameters
        rho = Constant(self.rho())

        for i in range(self.N):
            a = (1/rho)*inner(self.F*m, mv)*dx
            L = inner(-self.J*self.K()*inv(self.F.T)*grad(self.p[i]), mv)*dx

            solve(a == L, self.Uf[i], solver_parameters={"linear_solver": "minres",
                                                    "preconditioner": "hypre_amg"})

    def move_mesh(self):
        dU, L = self.Us.split(True)
        ALE.move(self.mesh, project(dU, VectorFunctionSpace(self.mesh, 'P', 1)))


    def choose_solver(self, prob):
        if self.params['Simulation']['solver'] == 'direct':
            return self.direct_solver(prob)
        else:
            return self.iterative_solver(prob)


    def solve(self):
        comm = mpi_comm_world()
        mpiRank = MPI.rank(comm)

        tol = self.TOL()
        maxiter = 100
        t = 0.0
        dt = self.dt()

        mprob = NonlinearVariationalProblem(self.MForm, self.mf, bcs=self.fbcs,
                                            J=self.dMForm)
        msol = self.choose_solver(mprob)

        sprob = NonlinearVariationalProblem(self.SForm, self.Us, bcs=self.sbcs,
                                            J=self.dSForm)
        ssol = self.choose_solver(sprob)

        while t < self.params['Parameter']['tf']:

            if mpiRank == 0: utils.print_time(t)

            for con in self.tconditions:
                con.t = t

            iter = 0
            eps = 1
            mf_ = Function(self.FS_F)
            while eps > tol and iter < maxiter:
                mf_.assign(self.p[0])
                ssol.solve()
                #sys.exit()
                self.fluid_solid_coupling()
                msol.solve()
                e = self.p[0] - mf_
                eps = np.sqrt(assemble(e**2*dx))
                iter += 1

            # Store current solution as previous
            self.mf_n.assign(self.mf)
            self.Us_n.assign(self.Us)

            # Calculate fluid vector
            if self.params['Parameter']['dt'] == self.params['Parameter']['tf']:
                print("\n ****************************************\n\
                Flow vector can only be calculated if dt!= tf \n\
                Simulation will continue without calculating the flow vector.\
                \n **************************************** ")

            elif (self.params['Parameter']['qo'] == 0.0) and (self.params['Parameter']['qi'] == 0.0):
                print("\n ****************************************\n\
                Flow vector can only be calculated if qi,qo!=0 ! \n\
                Simulation will continue without calculating the flow vector.\
                \n **************************************** ")

            else:
                self.calculate_flow_vector()

            # transform mf into list
            if self.N > 1:
                mf_list = [self.mf.sub(i) for i in range(self.N)]

            if self.N ==1:
                yield self.mf, self.Uf, self.p, self.Us, t
            else:
                yield mf_list, self.Uf, self.p, self.Us, t

            self.move_mesh()

            t += dt

        # Add a last print so that next output won't overwrite my time print statements
        print()


    def direct_solver(self, prob):
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'mumps'
        sol.parameters['newton_solver']['lu_solver']['reuse_factorization'] = True
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def iterative_solver(self, prob):
        TOL = self.TOL()
        sol = NonlinearVariationalSolver(prob)
        sol.parameters['newton_solver']['linear_solver'] = 'minres'
        sol.parameters['newton_solver']['preconditioner'] = 'hypre_amg'
        sol.parameters['newton_solver']['absolute_tolerance'] = TOL
        sol.parameters['newton_solver']['relative_tolerance'] = TOL*1e3
        sol.parameters['newton_solver']['maximum_iterations'] = 1000
        return sol


    def rho(self):
        return Constant(self.params['Parameter']['rho'])

    def phi(self):
        return Constant(self.params['Parameter']['phi'])

    def beta(self):
        beta = self.params['Parameter']['beta']
        if isinstance(beta, float):
            beta = [beta]
        return [Constant(b) for b in beta]

    def q_out(self):
        if isinstance(self.params['Parameter']['qo'], str):
            return Expression(self.params['Parameter']['qo'], degree=1)
        else:
            return Constant(self.params['Parameter']['qo'])

    # def q_in(self):
    #     class Qin(Expression):
    #         def __init__(self, territories, qin, **kwargs):
    #             self.territories = territories
    #             self.qin = qin
    #
    #         def eval_cell(self, values, x, cell):
    #             t = self.territories[cell.index]
    #             values[0] = self.qin[t] * (1 - exp(-pow(x[1], 2)/(2*pow(1.5, 2)))/(sqrt(2*pi)*1.5) * exp(-pow(x[2], 2)/(2*pow(1.5, 2)))/(sqrt(2*pi)*1.5))
    #
    #     qin = self.params.params['qi']
    #     if not isinstance(qin, list):
    #         qin = [qin]
    #
    #     q = Qin(self.territories, qin, degree=0)
    #     return q

    def q_in(self):
        if isinstance(self.params['Parameter']['qi'], str):
            return Expression(self.params['Parameter']['qi'], degree=1)
        else:
            return Constant(self.params['Parameter']['qi'])

    def K(self):
        #if self.N == 1:
        d = self.mf.geometric_dimension()
        I = Identity(d)
        K = Constant(self.params['Parameter']['K'])
        if self.fibers:
            return K*I
        else:
            return K*I
        #return K*I
        # else:
        #     d = self.u[0].geometric_dimension()
        #     I = Identity(d)
        #     K = [Constant(k) for k in self.params.params['K']]
        #     if self.fiber:
        #         return [k*self.fiber*I for k in K]
        #     else:
        #         return [k*I for k in K]

    def dt(self):
        return self.params['Parameter']['dt']

    def theta(self):
        theta = self.params['Parameter']['theta']
        return Constant(theta), Constant(1-theta)

    def TOL(self):
        return self.params['Parameter']['TOL']
