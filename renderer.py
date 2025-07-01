import h5py # reading .hdf5 files
import sys # for command line arguments
import numpy as np # array manipulation
import pyvista as pv # high level wrapper around VTK for building and rendering 3d meshes

def plot_mesh(mesh):
    # Render & save screenshot
    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True) # MAKES HEADLESS
    plotter.add_mesh(mesh, color="white")
    plotter.set_background("black")
    plotter.camera_position = "iso" # can be "xy" "zy" or a point
    plotter.line_smoothing = True
    # might want to use image_scale at some point to make images smaller or larger
    plotter.show(screenshot="output.png")

    return

def parse_mesh(filename):
    with h5py.File(filename, "r") as f:
        mesh_root = f["parts"]["part_001"]["mesh"] # contains subgroups 000, 001, ... which each contain one small mesh

        # Load each sub-mesh into a PolyData and collect them
        polys = []
        for mesh_id in sorted(mesh_root.keys(), key=int): # get names like 000, 001, ... and sort them numerically
            grp = mesh_root[mesh_id]
            pts = grp["points"][...]       # shape (N,3), absolute coords of each vertex
            tris = grp["triangle"][...]    # shape (M,3), how those vertices connect to form triangles, each value is a vertex's index in the pts array. 

                    # VTK wants faces in "flat" format: [3, i0, i1, i2,  3, j0, j1, j2, ...]. prefixing each triangle with a 3.
            faces = np.hstack([
                np.concatenate([[3], tri.astype(np.int64)])
                for tri in tris
            ]) # loop over each triangle, prepend the count 3, cast to int64, then horizontally stack them all into one flat array

            poly = pv.PolyData(pts, faces) # builds PolyData objects, which are the small meshes in parts 000, 001, ...
            polys.append(poly)

    # Merge all sub-meshes into one
    mesh = polys[0]
    for poly in polys[1:]:
        mesh = mesh.merge(poly)

    return mesh


def main(argc, argv):
    if argc != 2:
        print("please specify an argument.")
        sys.exit(1)

    mesh = parse_mesh(argv[1])
    plot_mesh(mesh)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)