import dolfin as df
import perspect
from geometry import Geometry, MarkerFunctions, Microstructure

comm = df.MPI.comm_world

def manufactured_solution(mesh, rho):
    from sympy import sin, cos
    import sympy as sym
    # Set up manufactured solution
    x, y, z = sym.symbols('x[0], x[1], x[2]')
    H = lambda x: sin(x) #* sin(y) * sin(z)
    F = lambda x: cos(x) #* cos(y) * cos(z)

    # define pressure
    p = 1-x**2
    p_code = sym.printing.ccode(p)
    print('C code for p:', p_code)
    p_D = df.Expression(p_code, domain=mesh, degree=2)

    # define solution
    m = 1-x**2
    m_code = sym.printing.ccode(m)
    print('C code for m:', m_code)
    m_D = df.Expression(m_code, domain=mesh, degree=2)

    # define permeability tensor
    k00 = 1
    k01 = k10 = 0
    k02 = k20 = 0
    k11 = 1
    k12 = k21 = 0
    k22 = 1

    # calculate source term
    gradp_x = sym.diff(p, x)
    gradp_y = sym.diff(p, y)
    gradp_z = sym.diff(p, z)
    kgradp_x = -k00*gradp_x - k01*gradp_y - k02*gradp_z
    kgradp_y = -k10*gradp_x - k11*gradp_y - k12*gradp_z
    kgradp_z = -k20*gradp_x - k21*gradp_y - k22*gradp_z
    divkp = sym.diff(kgradp_x, x) + sym.diff(kgradp_y, y) + sym.diff(kgradp_z, z)
    print(divkp)
    q = m/rho + divkp
    q = sym.simplify(q)
    q_code = sym.printing.ccode(q)
    print('C code for q:', q_code)
    q_D = df.Expression(q_code, domain=mesh, degree=2)

    return p_D, m_D, q_D


def run_perspect(mesh, qi, p_D, rho):

    # define boundaries
    class Base(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 0)

    class Endo(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 0)

    class Epi(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 1.0)

    markers = {'BASE': (10, 1),
            'ENDO': (30, 1),
            'EPI': (40, 1),
            'NONE': (0, 2)}
    ffun = df.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    ffun.set_all(markers['NONE'][0])
    base = Base()
    base.mark(ffun, markers['BASE'][0])
    endo = Endo()
    endo.mark(ffun, markers['ENDO'][0])
    epi = Epi()
    epi.mark(ffun, markers['EPI'][0])
    markerfunctions = MarkerFunctions(ffun=ffun)

    # define fiber structure
    VFS = df.VectorFunctionSpace(mesh, 'P', 1)
    f0 = df.interpolate(df.Constant((1.0, 0.0, 0.0)), VFS)
    s0 = df.interpolate(df.Constant((0.0, 1.0, 0.0)), VFS)
    n0 = df.interpolate(df.Constant((0.0, 0.0, 1.0)), VFS)
    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    # create geometry
    geometry = Geometry(mesh, markers=markers, markerfunctions=markerfunctions,
                        microstructure=microstructure)

    # Setup simulation
    parameters = {'mechanics': False, 'K': [1], 'qi': qi, 'qo': 0, 'dt': 1.0,
                    'theta': 1, 'steps': 1, 'rho': rho}
    pspect = perspect.Perspect(geometry, parameters=parameters)
    pspect.pprob.prescribed_pressure(p_D)
    pspect.solve_porous()

    m = pspect.pprob.state
    p = pspect.pprob.pressure
    u = pspect.pprob.darcy_flow

    return m, p, u


def test_mms0():
    nx = 10
    ny = 10
    nz = 10
    rho = 1
    mesh = df.UnitCubeMesh(nx, ny, nz)

    # load mms
    p_D, m_D, q_D = manufactured_solution(mesh, rho)

    # load perspect solution
    m, p, u = run_perspect(mesh, q_D, p_D, rho)

    print(df.assemble(m*df.dx), df.assemble(m_D*df.dx))

    mfile = df.XDMFFile(comm, "mms0.xdmf")
    mfile.parameters["flush_output"] = True
    mfile.parameters["rewrite_function_mesh"] = True
    mfile.parameters["functions_share_mesh"] = True
    mfile.write(m, 0)
    mfile.write(p, 0)
    mfile.write(u[0], 0)
    # mfile.write(q_D, 0)
    mfile.close()

    print(df.errornorm(m_D, m, mesh=mesh))
    print(mesh.hmin()**2)
    # assert df.errornorm(m, m_D, mesh=mesh) < 1e-5


if __name__ == '__main__':
    test_mms0()