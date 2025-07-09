import h5py # reading .hdf5 files
import sys # for command line arguments
import numpy as np # array manipulation
import pyvista as pv # high level wrapper around VTK for building and rendering 3d meshes

import os
import glob
import random


from plotting_utils import plot_meshes, plot_meshes_depth, plot_mesh
from spatial_utils import overlap, euclidean_distance
'''
Things to implement:
- randomized lighting
    - position, quantity, specular characteristics
- randomized camera poses
- LOOK INTO BLENDERPROC
- depth image DONE
- domain randomization
- image level augmentation
- in plane rotation DONE
- multiple objects in a scene together DONE
'''

# parses individual mesh
def parse_mesh(filename):
    """
    Open a .hdf5, look under every part_x/mesh subgroup, build Polydatas,
    and merge them into one. Raises if it finds no valid sub-mesh.
    """
    with h5py.File(filename, "r") as f:
        parts_grp = f.get("parts")
        if parts_grp is None:
            raise ValueError(f"{filename!r} has no top-level 'parts' group")

        polys = []
        # loop over every part (part_000, part_001, etc.)
        for part_name, part_grp in parts_grp.items():
            mesh_grp = part_grp.get("mesh")
            if mesh_grp is None:
                # no mesh in this part, skip
                continue

            # loop over every sub-mesh in this mesh group
            for mesh_id, sub in mesh_grp.items():
                pts = sub.get("points")
                tris = sub.get("triangle")
                if pts is None or tris is None:
                    continue

                pts = pts[...]
                tris = tris[...]

                # skip empty
                if pts.size == 0 or tris.size == 0:
                    continue

                # build faces: shape (M,4) of [3, i0, i1, i2], then flatten
                tris_int = tris.astype(np.int64)
                counts   = np.full((tris_int.shape[0], 1), 3, np.int64)
                faces    = np.hstack((counts, tris_int)).ravel()

                poly = pv.PolyData(pts, faces)
                polys.append(poly)

        if not polys:
            raise ValueError(f"No valid sub-meshes found in {filename!r}")

        # merge them all
        mesh = polys[0]
        for poly in polys[1:]:
            mesh = mesh.merge(poly)

        return mesh


# populates the meshes array to be used for images with multiple objects
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

# transforms meshes randomly in the scene.
def transform_meshes(meshes, translate_range=(0,200), min_sep=0.0, max_sep=50, max_trials=1000):
    placed_bounds = []
    transformed_meshes = []

    for mesh in meshes:
        orig_bounds = mesh.bounds
        last_tx, last_ty, last_tz = 0.0, 0.0, 0.0
        for _ in range(max_trials):
            tx = random.uniform(*translate_range)
            ty = random.uniform(*translate_range)
            tz = random.uniform(*translate_range)
            last_tx, last_ty, last_tz = tx, ty, tz

            new_bounds = (
                orig_bounds[0] + tx, orig_bounds[1] + tx,
                orig_bounds[2] + ty, orig_bounds[3] + ty,
                orig_bounds[4] + tz, orig_bounds[5] + tz
            )
            
            # randomly rotates the object on all 3 axis
            mesh.rotate_x(random.randint(0,360), inplace=True)
            mesh.rotate_y(random.randint(0,360), inplace=True)
            mesh.rotate_z(random.randint(0,360), inplace=True)
            
            # enforce max_sep
            if max_sep is not None and placed_bounds:
                # ensure new mesh is not too far from any existing one
                dists = [euclidean_distance(new_bounds, b) for b in placed_bounds]
                if any(dist > max_sep for dist in dists):
                    continue  # too far, retry


            # if new_bounds for mesh doesn't overlap other objects already placed, then add to new_bounds
            if all(not overlap(new_bounds, b, min_sep) for b in placed_bounds):
                placed_bounds.append(new_bounds)
                moved = mesh.copy().translate([tx, ty, tz], inplace=False)
                transformed_meshes.append(moved)
                break
        else: # if run out of trials for not overlapping, then accept overlap
            moved = mesh.copy().translate([last_tx, last_ty, last_tz], inplace=False)
            transformed_meshes.append(moved)
    
    return transformed_meshes

def abc(input_dir, output_dir):
    meshes = {}
    pattern = os.path.join(input_dir, "*", "*.hdf5")

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory.")
        sys.exit(1)

    for h5_path in glob.glob(pattern):
        # random sampling to keep it small
        if random.randint(1, 200) != 1:
            continue

        name = os.path.splitext(os.path.basename(h5_path))[0]
        mesh = parse_mesh(h5_path)
        meshes[name] = mesh

    # render each mesh into output_dir
    for name, mesh in meshes.items():
        print("currently rendering:", name)
        # build the output filepath
        out_png = os.path.join(output_dir, f"{name}.png")
        # assuming your plot_mesh takes (mesh, output_path)
        plot_mesh(mesh, out_png)

    return
    

# TODO: FIX SOME THINGS NOT WORKING:
# can't run scene render on abc/ input. only on input/ (python3 renderer.py abc 1 0/1)
# can't run single render on input/. only on abc/ (python3 renderer.py input 0 0/1)
# works: python3 renderer.py input 1 0/1
# works: python3 renderer.py abc 0 0/1
def main(argc, argv):
    if argc != 4:
        print("please specify an input directory as an argument, 0 or 1 for single or scene image, and 0 or 1 for rgb or depth.")
        sys.exit(1)

    orig_dir = os.getcwd()
    input_dir = argv[1]
    output_dir = os.path.join(orig_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    is_depth = int(argv[3])
    is_scene = int(argv[2])
    if is_scene == 1:
        meshes = make_meshes(orig_dir, input_dir)
        transformed_meshes = transform_meshes(meshes)
        if is_depth == 0:
            plot_meshes(transformed_meshes)
        else:
            plot_meshes_depth(transformed_meshes)
    else:
        abc(input_dir, output_dir)
    

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)