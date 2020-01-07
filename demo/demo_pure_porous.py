import dolfin as df
import numpy as np
import perspect
import pulse
import sys
import time
from geometry import Geometry, MarkerFunctions, Microstructure


comm = df.MPI.comm_world
df.set_log_level(40)

mesh = df.BoxMesh(comm, df.Point(0, 0, 0),
                    df.Point(3, 1, 0.1),
                    30, 10, 1)

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

VFS = df.VectorFunctionSpace(mesh, 'P', 1)
f0 = df.interpolate(df.Constant((0.5, 0.5, 0.0)), VFS)
s0 = df.interpolate(df.Constant((0.5, 0.0, 0.5)), VFS)
n0 = df.interpolate(df.Constant((0.0, 0.5, 0.5)), VFS)
microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

# load geometry
geometry = Geometry(mesh, markers=markers, markerfunctions=markerfunctions,
                    microstructure=microstructure)


qi = 0
pp = df.Expression('1-(x[0]*x[0]*x[0])', degree=1)
dt = 1e-3
parameters = {'mechanics': False, 'K': [1], 'qi': qi, 'qo': 0, 'dt': dt}

activation = df.Function(df.FunctionSpace(geometry.mesh, "R", 0))
activation.assign(df.Constant(0.0))
matparams = matparams = pulse.LinearElastic.default_parameters()
# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = 0.5*E/(1+nu), E*nu/((1+nu)*(1-2*nu))
matparams.update({'mu': mu, 'lmbda': lmbda})
material = pulse.LinearElastic(activation=activation,
                                parameters=matparams,
                                active_model="active_strain",
                                f0=geometry.f0,
                                s0=geometry.s0,
                                n0=geometry.n0)

pspect = perspect.Perspect(geometry, material, parameters=parameters)

us = []
m = []
m_volumes = []

def xdmf_parameters(xdmf_file):
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False
    xdmf_file.parameters["functions_share_mesh"] = True

mfile = df.XDMFFile(comm, "pure_porous.xdmf")
xdmf_parameters(mfile)

pspect.pprob.prescribed_pressure(pp)

t = 0
tf = 1
while t < tf:
    pspect.solve_porous()

    m = pspect.pprob.state
    p = pspect.pprob.pressure
    u = pspect.pprob.darcy_flow

    t += dt

    if (t/dt) % 100 < 1:
        mfile.write(m, t)
        mfile.write(p, t)
        mfile.write(u[0], t)

mfile.close()