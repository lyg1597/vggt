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
    extrinsics = data.get('extrinsics')

    # Check for required data
    if point_cloud_vertices is None or extrinsics is None:
        print("Error: The .npz file must contain 'point_cloud' and 'extrinsics' arrays.")
        return

    # Check for optional color data
    point_cloud_colors = data.get('point_cloud_colors')
    if point_cloud_colors is not None:
        print(f"Found {len(point_cloud_colors)} color entries.")
    else:
        print("No color data found for point cloud.")

    print(f"Found {len(point_cloud_vertices)} points and {len(extrinsics)} camera poses.")

    # --- 2. Create PyVista Plotter ---
    plotter = pv.Plotter(window_size=[1200, 800])

    # --- 3. Add Point Cloud to Scene ---
    if point_cloud_vertices.size > 0:
        cloud = pv.PolyData(point_cloud_vertices)
        
        actor_params = {
            "render_points_as_spheres": True,
            "point_size": 3,
        }

        if point_cloud_colors is not None and len(point_cloud_colors) == len(point_cloud_vertices):
            cloud["colors"] = point_cloud_colors
            actor_params["scalars"] = "colors"
            actor_params["rgb"] = True
        else:
            actor_params["color"] = "tan"

        plotter.add_mesh(cloud, **actor_params)
        
        # Calculate a reasonable scale for the axes based on the point cloud size
        scene_size = np.linalg.norm(cloud.bounds[1::2] - cloud.bounds[0::2])
        dynamic_axis_scale = scene_size * axis_scale
    else:
        print("Warning: Point cloud is empty.")
        dynamic_axis_scale = 1.0 # Default scale if there's no point cloud


    # --- 4. Add Camera Poses to Scene ---
    for i, E in enumerate(extrinsics):
        # The extrinsic matrix E maps world -> camera. We need the inverse (pose)
        # to place the camera in the world.
        
        # Create a 4x4 homogeneous world-to-camera matrix
        world_to_cam = np.eye(4)
        world_to_cam[:3, :] = E

        # Invert to get the camera pose (camera-to-world)
        cam_pose = np.linalg.inv(world_to_cam)

        # Create a small XYZ axis marker
        axes = pv.create_axes_marker(line_width=3, label_size=(0.0, 0.0))

        # Scale and position the axes using the camera pose
        axes.transform(cam_pose)
        axes.scale([dynamic_axis_scale] * 3, inplace=True)
        
        plotter.add_mesh(axes)

    # --- 5. Customize and Show Plot ---
    plotter.add_axes()
    plotter.enable_eye_dome_lighting()
    plotter.set_background('black')
    
    print("\nDisplaying scene. Close the window to exit.")
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud and camera poses from an .npz file.")
    parser.add_argument("npz_file", type=str, help="Path to the input .npz file created by the computation script.")
    parser.add_argument("--axis_scale", type=float, default=0.05, help="Scale of the camera axis markers relative to the scene size.")
    
    args = parser.parse_args()
    visualize_scene(args.npz_file, args.axis_scale)
