import perspect
import dolfin as df

def test_perspect():
    geometry = perspect.HeartGeometry.from_file(perspect.mesh_paths["simple_ellipsoid"])

    p = perspect.Perspect(geometry)
    assert 1==1
