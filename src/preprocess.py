import numpy as np
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
from skimage import measure


VOXEL_RESOLUTION = 128

def create_bounds(mesh):
    pos = np.array(mesh.vertices)

    # Get the bottom left corner and top right
    x_min, y_min, z_min = np.array((pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min()))
    x_max, y_max, z_max = np.array((pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max()))

    # Create a box of points
    xs = np.linspace(x_min, x_max, num=VOXEL_RESOLUTION)
    ys = np.linspace(y_min, y_max, num=VOXEL_RESOLUTION)
    zs = np.linspace(z_min, z_max, num=VOXEL_RESOLUTION)
    
    # Generate a 3D cube aorund object
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.moveaxis(np.stack([X, Y, Z]), 0, -1)

    return points

def load_mesh(path):
    mesh = trimesh.load("resources/obj/jug02.obj")

    bounds = create_bounds(mesh)

    return bounds


def ray_trace(O, D, V0, V1, V2, eps=1e-8):
    E_1 = V1 - V0
    E_2 = V2 - V0

    # Declare Matrices
    A = np.array([D, -E_1, -E_2], dtype=float).T
    B = (V0 - O).astype(float)

    if abs(np.linalg.det(A)) < eps:
        return False

    t, u, v = np.linalg.solve(A, B)

    return (t >= 0) and (u>= 0) and (v >= 0) and (u + v <= 1)




if __name__ == "__main__":
    mesh = trimesh.load("resources/obj/jug02.obj")
    print(f"Loaded mesh: {mesh}")
    print("Generating SDF and visualizing...")
    

    O = np.array([0, 0, 0])
    D = np.array([1, 0, 0])
    
    V0 = np.array([1, 1, 0])
    V1 = np.array([1, -1, 0])
    V2 = np.array([1, 0, 1])

    print(ray_trace(O, D, V0, V1, V2))


