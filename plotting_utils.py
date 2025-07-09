import pyvista as pv # high level wrapper around VTK for building and rendering 3d meshes
import numpy as np # array manipulation
import matplotlib.pyplot as plt

# plots multiple meshes to output.png
def plot_meshes(meshes):
    # Render & save screenshot
    # pv.start_xvfb() # UNCOMMENT FOR RUNNING ON COMPUTING CLUSTER
    plotter = pv.Plotter(off_screen=True) # MAKES HEADLESS

    for mesh in meshes:
        plotter.add_mesh(mesh, color="00ff60")

    # plotter.set_background("black")
    plotter.camera_position = "iso" # can be "xy" "zy" or a point
    plotter.line_smoothing = True
    # might want to use image_scale at some point to make images smaller or larger
    plotter.show(auto_close=False)
    plotter.screenshot("output.png", transparent_background=True)
    plotter.close()
    return

# plots multiple meshes as depth image to output.png
def plot_meshes_depth(meshes):
    # pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True)

    for mesh in meshes:
        plotter.add_mesh(mesh, color=True)
    
    plotter.camera_position = "iso"
    plotter.line_smoothing = True
    plotter.image_store = True

    plotter.show(auto_close=False)
    depth = plotter.get_image_depth()
    plotter.close()

    # replace NaNs, normalize to [0,1]
    depth = np.nan_to_num(depth, nan=0.0)
    dmin, dmax = depth.min(), depth.ptp() or 1.0
    norm = (depth - dmin) / dmax
    # save as 8-bit grayscale PNG (nearer = brighter)
    depth_img = (255 * (1.0 - norm)).astype(np.uint8)

    plt.imsave("output.png", depth_img, cmap="gray", vmin=0, vmax=255)
    return

def plot_mesh(mesh, output_path):
    # pv.start_xvfb() # UNCOMMENT FOR RUNNING ON COMPUTING CLUSTER
    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(mesh, color="00ff60")

    plotter.set_background("black")
    plotter.camera_position = "iso" # can be "xy" "zy" or a point
    plotter.line_smoothing = True

    plotter.show(screenshot=output_path)
    return