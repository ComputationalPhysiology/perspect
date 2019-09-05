
from perspect.perspect import Perspect

import dolfin as df

from pulse.geometry import Geometry, HeartGeometry
from pulse.example_meshes import mesh_paths
from pulse.utils import get_lv_marker


# Solver parameters
flags = ["-O3", "-ffast-math", "-march=native"]
df.parameters["form_compiler"]["quadrature_degree"] = 4
df.parameters["form_compiler"]["representation"] = "uflacs"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
df.parameters["allow_extrapolation"] = True
