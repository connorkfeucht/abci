import h5py # reading .hdf5 files
import sys # for command line arguments
import numpy as np # array manipulation
import pyvista as pv # high level wrapper around VTK for building and rendering 3d meshes

import os
import glob
import random

def plot_meshes(meshes):
    # Render & save screenshot
    # pv.start_xvfb() # UNCOMMENT FOR RUNNING ON COMPUTING CLUSTER
    plotter = pv.Plotter(off_screen=True) # MAKES HEADLESS

    for mesh in meshes:
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

def make_meshes(orig_dir, input_dir):
    meshes = []
    if not os.path.isdir(input_dir):
        print(f"Error: “{input_dir}” is not a directory.")
        sys.exit(1)

    os.chdir(input_dir)

    for hdf5_file in glob.glob("*.hdf5"):
        mesh = parse_mesh(hdf5_file)
        meshes.append(mesh)

    os.chdir(orig_dir)
    return meshes

def transform_meshes(meshes): # TODO: make it so that meshes cannot render inside eachother
    transformed_meshes = [mesh.translate([random.randint(1,300), random.randint(1,300), random.randint(1,300)]) for mesh in meshes]
    return transformed_meshes


def main(argc, argv):
    if argc != 2:
        print("please specify an input directory as an argument.")
        sys.exit(1)

    orig_dir = os.getcwd()
    input_dir = argv[1]
    meshes = make_meshes(orig_dir, input_dir)
    transformed_meshes = transform_meshes(meshes)
    plot_meshes(transformed_meshes)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)