import dolfin as df
import mshr

from pulse.geometry_utils import generate_fibers
from geometry import Microstructure, Geometry, MarkerFunctions


comm = df.MPI.comm_world

# get mesh from Gmsh
mesh = df.Mesh()
fmesh = df.XDMFFile(comm, "./biv_meshes/biv20k.xdmf")
fmesh.read(mesh)
fmesh.close()

base_x = 0.0

# LV
center_lv = df.Point(0.0, 0.0, 0.0)
a_lv_epi = 12.0
b_lv_epi = 6.0
c_lv_epi = 6.0
a_lv_endo = 9.0
b_lv_endo = 3.0
c_lv_endo = 3.0


# RV
center_rv = df.Point(0.0, 3.0, 0.0)
a_rv_epi = 10.5
b_rv_epi = 9.0
c_rv_epi = 6.0
a_rv_endo = 8.7
b_rv_endo = 7.5
c_rv_endo = 4.5

## Markers
base_marker = 10
endolv_marker = 30

epi_marker = 40
markers = dict(BASE=(10, 2),
               ENDO_RV = (20, 2),
               ENDO_LV=(30, 2),
               EPI=(40, 2))


class EndoLV(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center_lv.x())**2/a_lv_endo**2 \
            + (x[1]-center_lv.y())**2/b_lv_endo**2 \
            + (x[2]-center_lv.z())**2/c_lv_endo**2 - 1.1 < df.DOLFIN_EPS and on_boundary

class Base(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] + df.DOLFIN_EPS > 0.0 and on_boundary

class EndoRV(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]-center_rv.x())**2/a_rv_endo**2 \
            + (x[1]-center_rv.y())**2/b_rv_endo**2 \
            + (x[2]-center_rv.z())**2/c_rv_endo**2 - 1.1 < df.DOLFIN_EPS   \
            and (x[0]-center_lv.x())**2/a_lv_epi**2 \
            + (x[1]-center_lv.y())**2/b_lv_epi**2 \
            + (x[2]-center_lv.z())**2/c_lv_epi**2 - 0.9 > df.DOLFIN_EPS) and on_boundary

class Epi(df.SubDomain):
    def inside(self, x, on_boundary):
        return (x[0]-center_rv.x())**2/a_rv_epi**2 \
            + (x[1]-center_rv.y())**2/b_rv_epi**2 \
            + (x[2]-center_rv.z())**2/c_rv_epi**2 - 0.9 > df.DOLFIN_EPS   \
            and (x[0]-center_lv.x())**2/a_lv_epi**2 \
            + (x[1]-center_lv.y())**2/b_lv_epi**2 \
            + (x[2]-center_lv.z())**2/c_lv_epi**2 - 0.9 > df.DOLFIN_EPS and on_boundary


# Create facet function
ffun = df.MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

endolv = EndoLV()
endolv.mark(ffun, markers['ENDO_LV'][0])
endorv = EndoRV()
endorv.mark(ffun, markers['ENDO_RV'][0])
epi = Epi()
epi.mark(ffun, markers['EPI'][0])
base = Base()
base.mark(ffun, markers['BASE'][0])

# Create cell function
cfun = df.MeshFunction("size_t", mesh, 3)
cfun.set_all(0)

# Mark mesh
for facet in df.facets(mesh):
    mesh.domains().set_marker((facet.index(), ffun[facet]), 2)

marker_functions = MarkerFunctions(ffun=ffun, cfun=cfun)

# Save to xdmf to check out in paraview
# xdmf = df.XDMFFile(df.MPI.comm_world, "biv_geometry_markers.xdmf")
# xdmf.write(ffun)
# xdmf.close()


# Make fiber field
fiber_params = df.Parameters("Fibers")
fiber_params.add("fiber_space", "CG_1")
# fiber_params.add("fiber_space", "Quadrature_4")
fiber_params.add("include_sheets", False)
fiber_params.add("fiber_angle_epi", -60)
fiber_params.add("fiber_angle_endo", 60)

try:
    fields = generate_fibers(mesh, fiber_params)
except ImportError:
    fields = []
    fields_names = []
else:
    fields_names = ['f0', 's0', 'n0']

# Save to xdmf to check out in paraview
# xdmf = df.XDMFFile(df.MPI.comm_world, "biv_geometry_f0.xdmf")
# xdmf.write(fields[0])
# xdmf.close()

microstructure = Microstructure(**dict(zip(fields_names, fields)))

# Save to xdmf to check out in paraview
# xdmf = df.XDMFFile(df.MPI.comm_world, "biv_geometry_.xdmf")
# xdmf.write(mesh)
# xdmf.close()

geometry = Geometry(mesh, markers=markers,
                    markerfunctions=marker_functions,
                    microstructure=microstructure)
geometry.save('biv_geometry20k')
from IPython import embed; embed()