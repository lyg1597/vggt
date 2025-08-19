import numpy as np
import pyvista as pv
import argparse
import os
from scipy.spatial.transform import Rotation

def visualize_point_cloud(plotter, points, colors):
    """
    Creates an interactive 3D plot of the transformed point cloud.

    Args:
        points (np.ndarray): The point cloud in the JSON frame.
        colors (np.ndarray): The RGB colors for the points.
    """
    print("Opening visualization window...")
    cloud = pv.PolyData(points[::20,:].astype('float'))
    cloud["colors"] = colors[::20,:].astype('float')

    # plotter = pv.Plotter(window_size=[1200, 800])
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=5,
        scalars="colors",
        rgb=True,
        ambient=1.0,
        show_edges=False, 
        lighting=False,
    )
    plotter.add_title("Transformed Point Cloud (JSON Frame)", font_size=12)
    # plotter.enable_eye_dome_lighting()
    plotter.add_axes()
    print("Close the PyVista window to exit the script.")
    # plotter.show()
    return plotter


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

    point_cloud_vertices = data.get('raw_point_cloud').reshape((-1,3))
    point_cloud_color = data.get('raw_point_color').reshape((-1,3))
    if 'extrinsics' in data:
        extrinsics = data.get('extrinsics')
        world_to_camera = np.zeros((extrinsics.shape[0],4,4))
        world_to_camera[:,:3,:] = extrinsics 
        world_to_camera[:,3,3] = 1
        camera_to_world = np.linalg.inv(world_to_camera)
    else:
        camera_to_world = np.array([])
    
    p = pv.Plotter()
    p.background_color = "white"

    # Global XYZ reference axes (world frame)
    p.add_mesh(pv.Line((0, 0, 0), (2, 0, 0)), color="red",   line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, 2, 0)), color="green", line_width=4)
    p.add_mesh(pv.Line((0, 0, 0), (0, 0, 2)), color="blue",  line_width=4)

    p = visualize_point_cloud(p, point_cloud_vertices, point_cloud_color)
 
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
