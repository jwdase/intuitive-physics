from src.preprocess import fill_box
import trimesh


def main():
    mesh = trimesh.load("resources/obj/jug02.obj")
    print(f"Loaded mesh: {mesh}")
    print("Generating SDF and visualizing...")
    fill_box(mesh)


if __name__ == "__main__":
    main()
