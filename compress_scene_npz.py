import numpy as np
import open3d as o3d

def compress_data(combined_data, voxel_size=0.05):
    """
    Compresses and cleans a point cloud using outlier removal and voxel down-sampling.

    Args:
        combined_data (np.ndarray): The point cloud data to compress (Nx6 array).
        voxel_size (float): The voxel size used for initial registration, used to derive
                            the final compression voxel size.

    Returns:
        np.ndarray: The compressed and cleaned Nx6 NumPy array.
        o3d.geometry.PointCloud: The final Open3D point cloud object.
    """
    print("\n--- Compressing and Cleaning Point Cloud ---")
    
    # Convert numpy array to Open3D object
    pcd_to_compress = o3d.geometry.PointCloud()
    pcd_to_compress.points = o3d.utility.Vector3dVector(combined_data[:, :3])
    # Ensure colors are in the [0, 1] range for Open3D
    pcd_to_compress.colors = o3d.utility.Vector3dVector(combined_data[:, 3:])

    # a) Statistical Outlier Removal
    print("1. Removing statistical outliers...")
    cl, ind = pcd_to_compress.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_compressed = pcd_to_compress.select_by_index(ind)

    # b) Voxel Down-sampling
    final_voxel_size = voxel_size * 0.5
    print(f"2. Down-sampling final cloud with voxel size: {final_voxel_size}")
    pcd_compressed = pcd_compressed.voxel_down_sample(voxel_size=final_voxel_size)

    # Convert back to NumPy
    final_points = np.asarray(pcd_compressed.points)
    final_colors = np.asarray(pcd_compressed.colors)
    final_array = np.hstack((final_points, final_colors))

    print("\n--- Compression Complete ---")
    return final_array, pcd_compressed

# --- Main execution ---

# 1. Create Sample Data
# We'll generate a random N*6 point cloud to simulate your data.
# Note: Open3D expects RGB color values to be in the [0, 1] range.
# print("1. Generating sample point cloud data...")
# num_points = 100000
# # [x, y, z, r, g, b]
# point_cloud_data = np.random.rand(num_points, 6) 
# # Make the point cloud look more like a shape (e.g., a sphere)
# point_cloud_data[:, :3] = point_cloud_data[:, :3] - 0.5 
data = np.load('./scene_dist_0200-2.npz')
point_cloud_pos = data['point_cloud']
point_cloud_color = data['point_cloud_colors'].astype(float)/255.0
point_cloud_data = np.hstack((point_cloud_pos, point_cloud_color))

# 2. Use your function to compress the data
# The function returns both the NumPy array and the Open3D PointCloud object.
# We'll need the Open3D object for saving and visualization.
compressed_array, compressed_pcd = compress_data(point_cloud_data, voxel_size=0.05)
print(f"Original number of points: {len(point_cloud_data)}")
print(f"Compressed number of points: {len(compressed_array)}")

# 3. Save the compressed point cloud as a .ply file 💾
output_filename = "scene_dist_0200-2_compressed.ply"
print(f"\n3. Saving compressed point cloud to '{output_filename}'...")
o3d.io.write_point_cloud(output_filename, compressed_pcd)
print("File saved successfully.")

# 4. Visualize the compressed data as a Point Cloud 📊
print("\n4. Displaying the compressed point cloud. Close the window to continue.")
o3d.visualization.draw_geometries([compressed_pcd])

# 5. Visualize the compressed point cloud as a voxel grid 🧊
# The voxel size should correspond to the level of detail you want to see.
# Using the size from the compression step is a good starting point.
voxel_size_for_vis = 0.05 * 0.5 
print(f"\n4. Creating and visualizing a voxel grid with voxel size {voxel_size_for_vis}...")

# Create a VoxelGrid object from the compressed point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    compressed_pcd,
    voxel_size=voxel_size_for_vis
)

# Visualize the voxel grid. This will open an interactive window.
print("Displaying voxel grid. Close the window to exit.")
o3d.visualization.draw_geometries([voxel_grid])