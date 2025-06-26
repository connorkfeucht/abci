import sys
import torch, pytorch3d
import os
import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader, PointLights
)
from pytorch3d.structures import Meshes


def build_renderer(device):
    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(2.7, 10, 20)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1,)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. Here we can use a predefined
    # PhongShader, passing in the device on which to initialize the default parameters
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights))
    
    return renderer

def load_mesh(filename, device):
    # open and locate the mesh groups 
    verts_list = []
    faces_list = []
    with h5py.File(filename, "r") as f:
        mesh_root = f["parts"]["part_001"]["mesh"] # contains subgroups 000, 001, ... which each contain one small mesh

        # load each sub-mesh into a PolyData and collect them
        polys = []
        for mesh_id in sorted(mesh_root.keys(), key=int): # get names like 000, 001, ... and sort them numerically
            grp = mesh_root[mesh_id]
            pts = grp["points"][...]       # shape (N,3), absolute coords of each vertex
            tris = grp["triangle"][...]    # shape (M,3), how those vertices connect to form triangles, each value is a vertex's index in the pts array. 

            verts = torch.from_numpy(pts).float().to(device) # creates tensor, out of numpy Nx3 array, converts from float64 -> float32, sends to device (cpu)
            verts_list.append(verts)

            faces = torch.from_numpy(tris).float().to(device)
            faces_list.append(faces)
    
    mesh = Meshes(verts=verts_list, faces=faces_list) # creating one batched Meshes object containing all sub-meshes

    return mesh


def main(argc, argv):
    if argc != 2:
        print("please specify an argument.")
        return
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    mesh = load_mesh(argv[1], device)
    renderer = build_renderer(device)
    image = renderer(mesh)
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0].numpy())



if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
