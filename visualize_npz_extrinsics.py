import numpy as np
import pyvista as pv
import argparse
import os

def visualize_scene(npz_file_path: str, axis_scale: float = 0.1):
    """
    Loads a scene from an .npz file and visualizes the point cloud
    and camera extrinsics as XYZ axes.

    Args:
        npz_file_path (str): The path to the input .npz file.
        axis_scale (float): The scaling factor for the camera axis markers.
    """
    # --- 1. Load the Data ---
    if not os.path.exists(npz_file_path):
        print(f"Error: File not found at '{npz_file_path}'")
        return

    print(f"Loading scene data from {npz_file_path}...")
    data = np.load(npz_file_path)

    point_cloud_vertices = data.get('point_cloud')
    point_cloud_color = data.get('point_cloud_color')
    extrinsics = data.get('extrinsics')
    world_to_camera = np.zeros((extrinsics.shape[0],4,4))
    world_to_camera[:,:3,:] = extrinsics 
    world_to_camera[:,3,3] = 1
    camera_to_world = np.linalg.inv(world_to_camera)
    
    p = pv.Plotter()
    p.background_color = "white"

    # Global XYZ reference axes (world frame)
    p.add_mesh(pv.Line((0, 0, 0), (2, 0, 0)), color="red",   line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, 2, 0)), color="green", line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, 0, 2)), color="blue",  line_width=4)

    for i in range(camera_to_world.shape[0]):           # rgb_path unused here

        # Full camera rotation with –5° pitch offset applied
        pos = camera_to_world[i,:3,3]
        R_final = camera_to_world[i,:3,:3]

        # Local axes endpoints
        x_end = pos + R_final @ np.array([0.2, 0, 0])
        y_end = pos + R_final @ np.array([0, 0.2, 0])
        z_end = pos + R_final @ np.array([0, 0, 0.2])

        # Draw camera-frame axes (short lines)
        p.add_mesh(pv.Line(pos, x_end), color="red",   line_width=2)
        p.add_mesh(pv.Line(pos, y_end), color="green", line_width=2)
        p.add_mesh(pv.Line(pos, z_end), color="blue",  line_width=2)

        # Red point marking the camera position
        p.add_mesh(pv.Sphere(radius=0.01, center=pos), color="red")

    p.show(cpos="xy")                     # top-down view


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud and camera poses from an .npz file.")
    parser.add_argument("npz_file", type=str, help="Path to the input .npz file created by the computation script.")
    parser.add_argument("--axis_scale", type=float, default=0.05, help="Scale of the camera axis markers relative to the scene size.")
    
    args = parser.parse_args()
    visualize_scene(args.npz_file, args.axis_scale)
