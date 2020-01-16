import dolfin as df
import numpy as np
import perspect
from geometry import Geometry, MarkerFunctions, Microstructure

comm = df.MPI.comm_world

def manufactured_solution(rho):
    from sympy import sin, cos, exp, tanh
    import sympy as sym
    # Set up manufactured solution
    x, y, z = sym.symbols('x[0], x[1], x[2]')
    F = lambda x: sin(x)
    H = lambda x: tanh(x)

    # define pressure
    p = F(x*y)
    p_code = sym.printing.ccode(p)
    print('C code for p:', p_code)

    # define solution
    m = H(x*y)
    m_code = sym.printing.ccode(m)
    print('C code for m:', m_code)

    # calculate source term
    gradp_x = sym.diff(p, x)
    gradp_y = sym.diff(p, y)
    gradp_z = sym.diff(p, z)
    grad_code = sym.printing.ccode(gradp_x+gradp_y+gradp_z)
    print('C code for gradp:', grad_code)
    divkp = sym.diff(-rho*gradp_x, x) + sym.diff(-rho*gradp_y, y) +\
            sym.diff(-rho*gradp_z, z)
    div_code = sym.printing.ccode(divkp)
    print('C code for div:', div_code)
    q = m/rho + divkp/rho
    q = sym.simplify(q)
    q_code = sym.printing.ccode(q)
    print('C code for q:', q_code)

    return p_code, m_code, q_code


def run_perspect(mesh, qi, p_D, m_D, rho):
    # Define boundaries
    class Endo(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 0)

    markers = {'ENDO': (30, 1)}
    ffun = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    ffun.set_all(0)
    endo = Endo()
    endo.mark(ffun, markers['ENDO'][0])
    markerfunctions = MarkerFunctions(ffun=ffun)

    # define fiber structure
    VFS = df.VectorFunctionSpace(mesh, 'P', 1)
    f0 = df.interpolate(df.Constant((1.0, 0.0, 0.0)), VFS)
    s0 = df.interpolate(df.Constant((0.0, 1.0, 0.0)), VFS)
    n0 = df.interpolate(df.Constant((0.0, 0.0, 1.0)), VFS)
    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    # create geometry
    geometry = Geometry(mesh, markers=markers, microstructure=microstructure,
                        markerfunctions=markerfunctions)

    # Setup simulation
    parameters = {'mechanics': False, 'K': [1], 'qi': qi, 'qo': 0, 'dt': 1.0,
                    'theta': 1, 'steps': 1, 'rho': rho}
    solver_parameters = {'linear_solver': 'cg', 'preconditioner': 'sor'}
    pspect = perspect.Perspect(geometry, parameters=parameters,
                                solver_parameters=solver_parameters)
    bcs = [df.DirichletBC(pspect.pprob.state.function_space(), m_D, "on_boundary")]
    pspect.pprob.prescribed_pressure(p_D)
    pspect.solve_porous(bcs)

    m = pspect.pprob.state

    return m


def run_mms0():
    df.set_log_level(20)
    N = [2, 4, 8, 16, 32, 40, 50, 60, 70, 80]
    errors = []
    dxs = []
    
    rho = 0.7

    # load mms
    p_code, m_code, q_code = manufactured_solution(rho)

    for nx in N:
        mesh = df.UnitCubeMesh(nx, nx, nx)
        p_D = df.Expression(p_code, degree=2)
        m_D = df.Expression(m_code, degree=2)
        q_D = df.Expression(q_code, degree=2)

        # load perspect solution
        m = run_perspect(mesh, q_D, p_D, m_D, rho)

        e = df.errornorm(m_D, m, degree_rise=0)

        F = df.FunctionSpace(mesh, 'P', 2)
        n = df.interpolate(m_D, F)
        f = df.File("test.pvd")
        f << m
        errors.append(e)
        dxs.append(mesh.hmin())
    
    errors = np.array(errors)
    dxs = np.array(dxs)
    conv_rates = np.log(errors[1:]/errors[0:-1])/np.log(dxs[1:]/dxs[0:-1])

    print(errors)
    print(dxs)
    print(conv_rates)

    import matplotlib.pyplot as plt 
    plt.loglog(dxs, errors)
    plt.loglog(dxs, errors, marker='o')
    plt.xlabel("dx")
    plt.ylabel("L2 errornorm")
    plt.grid(True, which="both")
    plt.savefig("mms0_convergence.png")

if __name__ == '__main__':
    run_mms0()