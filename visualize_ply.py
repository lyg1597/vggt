import pyvista as pv
import os

# --- Configuration ---
ply_file_path = 'output_point_cloud.ply'

# --- Check if the file exists ---
if not os.path.exists(ply_file_path):
    print(f"Error: The file '{ply_file_path}' was not found.")
    print("Please make sure the file is in the same directory as the script, or provide the full path.")
else:
    # --- Step 1: Read the PLY file ---
    # PyVista's read() function automatically detects the file type
    # and loads the data, including vertex colors if they exist.
    point_cloud = pv.read(ply_file_path)
    print("Successfully loaded the PLY file.")
    print("\n--- Point Cloud Information ---")
    print(point_cloud)

    # --- Step 2: Create a plotter object ---
    # This is the main object for creating a scene and plotting meshes.
    plotter = pv.Plotter(window_size=[1000, 800])

    # --- Step 3: Add the point cloud to the plotter ---
    # We'll display the point cloud in a few different ways.
    # Choose the one that works best for you by uncommenting it.

    # Option A: Render points as spheres (looks good for sparse clouds)
    # `rgba=True` tells PyVista to use the RGBA color data from the file.
    plotter.add_mesh(
        point_cloud,
        render_points_as_spheres=True,
        point_size=1,
        scalars=point_cloud.point_data['RGBA'],
        rgb=True,
        ambient=1.0,
        show_edges=False, 
        lighting=False,
    )

    # # Option B: Render as simple points (more performant for dense clouds)
    # plotter.add_mesh(
    #     point_cloud,
    #     style='points',
    #     point_size=5,
    #     rgba=True
    # )
    
    # # Option C: Use Eye-Dome Lighting for better depth perception
    # # This is a shading technique that enhances the 3D structure.
    # plotter.add_mesh(
    #     point_cloud,
    #     style='points',
    #     point_size=5,
    #     rgba=True
    # )
    # plotter.enable_eye_dome_lighting()


    # --- Step 4: Customize and show the plot ---
    print("\nDisplaying the plot. Press 'q' to close the window.")

    # Add a helpful axes widget
    plotter.add_axes()

    # Improve the background color
    plotter.set_background('white')

    # Show the interactive window
    plotter.show()

    # except Exception as e:
    #     print(f"An error occurred while trying to visualize the file: {e}")