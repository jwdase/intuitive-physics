import numpy as np
import trimesh

def create_bounds(mesh):

    pos = np.array(mesh.vertices)

    # Get the bottom left corner and top right
    lower_corner = np.array((pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min()))
    upper_corner = np.array((pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max()))

    return np.stack([lower_corner, upper_corner])


def load_mesh(path):
    mesh = trimesh.load("resources/obj/jug02.obj")

    bounds = create_bounds(mesh)

    return bounds


if __name__ == "__main__":
    x = load_mesh("h")
