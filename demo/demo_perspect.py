import matplotlib.pyplot as plt
import dolfin as df
import pulse
import perspect
import sys

comm = df.mpi_comm_world()
df.set_log_level(16)

if len(sys.argv) > 1:
    gamma_space = sys.argv[1]
else:
    gamma_space = "R_0"

# load geometry
geometry = perspect.Geometry.from_file(perspect.mesh_paths["simple_ellipsoid"])

if gamma_space == "regional":
    activation = pulse.RegionalParameter(geometry.cfun)
    target_activation = pulse.dolfin_utils.get_constant(0.2, len(activation))
else:
    activation = df.Function(df.FunctionSpace(geometry.mesh, "R", 0))
    target_activation = df.Constant(0.2)

activation.assign(target_activation)
matparams = perspect.HolzapfelOgden.default_parameters()
material = perspect.HolzapfelOgden(
    activation=activation,
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)

# LV Pressure
lvp = df.Constant(1.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = perspect.mechanicsproblem.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Add spring term at the base with stiffness 1.0 kPa/cm^2
base_spring = 1.0
robin_bc = [
    perspect.mechanicsproblem.RobinBC(
        value=df.Constant(base_spring), marker=geometry.markers["BASE"][0]
    )
]

# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = df.DirichletBC(
        V.sub(0), df.Constant(0.0), geometry.ffun, geometry.markers["BASE"][0]
    )
    return bc


dirichlet_bc = [fix_basal_plane]

# Collect boundary conditions
bcs = perspect.mechanicsproblem.BoundaryConditions(
    dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
)

parameters = {'mechanics': True, 'qi': 1e-3}

p = perspect.Perspect(geometry, material, mechanics_bcs=bcs, parameters=parameters)

p.solve()

# Get the solution
u, x = p.mprob.state.split(deepcopy=True)
m = p.pprob.state

# Move mesh accoring to displacement
# u_int = df.interpolate(u, df.VectorFunctionSpace(geometry.mesh, "CG", 1))
# mesh = df.Mesh(geometry.mesh)
# df.ALE.move(mesh, u_int)

ufile = df.XDMFFile(comm, "u.xdmf")
mfile = df.XDMFFile(comm, "m.xdmf")
ufile.write(u, 0)
mfile.write(m, 0)
ufile.close()
mfile.close()
