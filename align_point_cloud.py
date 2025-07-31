import open3d as o3d
import numpy as np
import copy

def align_point_clouds_fast(pcd1_data, pcd2_data):
    """
    Aligns two point clouds using the Fast Global Registration (FGR) algorithm.

    FGR is a fast and accurate method for global registration that often eliminates
    the need for a separate ICP refinement step, or provides a very strong
    initial alignment for it.

    Args:
        pcd1_data (np.ndarray): The target point cloud data as an N1x6 NumPy array (X, Y, Z, R, G, B).
                                 Color values should be in the range [0, 1].
        pcd2_data (np.ndarray): The source point cloud data as an N2x6 NumPy array (X, Y, Z, R, G, B)
                                 that will be aligned to pcd1_data. Color values should be in the range [0, 1].

    Returns:
        np.ndarray: A combined (N1+N2)x6 NumPy array of the aligned point clouds.
        o3d.geometry.PointCloud: The combined and aligned Open3D point cloud object.
    """
    print("--- Starting Point Cloud Registration with Fast Global Registration (FGR) ---")

    # 1. Convert NumPy arrays to Open3D PointCloud objects
    # ---------------------------------------------------------
    print("1. Converting NumPy arrays to Open3D PointCloud objects...")
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pcd1_data[:, :3])
    target.colors = o3d.utility.Vector3dVector(pcd1_data[:, 3:])
    
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcd2_data[:, :3])
    source.colors = o3d.utility.Vector3dVector(pcd2_data[:, 3:])

    # --- KEY PARAMETER TO TUNE ---
    # Voxel size for downsampling. This is still the most critical parameter.
    voxel_size = 0.05 
    
    # 2. Prepare Point Clouds for FGR
    # ---------------------------------------------------------
    print("\n2. Downsampling and preparing features for FGR...")
    
    # Downsample point clouds to speed up feature computation
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 3. Perform Fast Global Registration
    # ---------------------------------------------------------
    print("\n3. Performing Fast Global Registration...")
    distance_threshold = voxel_size * 1.5
    fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    print(f"   FGR fitness: {fgr_result.fitness:.4f}")
    print(f"   FGR inlier_rmse: {fgr_result.inlier_rmse:.4f}")

    # Apply the FGR transformation to the original source cloud
    source.transform(fgr_result.transformation)
    
    # 4. Combine and Post-process Point Clouds
    # ---------------------------------------------------------
    print("\n4. Combining and post-processing point clouds...")
    combined_pcd = target + source

    # Optional: Voxel Down-sampling on the final cloud to merge and clean it
    final_voxel_size = voxel_size * 0.5
    print(f"   Down-sampling final cloud with voxel size: {final_voxel_size}")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=final_voxel_size)
    
    # 5. Convert Final Cloud to NumPy
    # ---------------------------------------------------------
    print("\n5. Converting final cloud to NumPy array...")
    combined_points = np.asarray(combined_pcd.points)
    combined_colors = np.asarray(combined_pcd.colors)
    
    final_aligned_array = np.hstack((combined_points, combined_colors))

    print("\n--- Registration Complete ---")
    
    return final_aligned_array, combined_pcd

def align_point_clouds_open3d(pcd1_data, pcd2_data):
    """
    Aligns two point clouds using a coarse-to-fine registration approach with Open3D.
    This version uses a more robust Point-to-Plane ICP for fine alignment to reduce ghosting.

    Args:
        pcd1_data (np.ndarray): The target point cloud data as an N1x6 NumPy array (X, Y, Z, R, G, B).
                                 Color values should be in the range [0, 1].
        pcd2_data (np.ndarray): The source point cloud data as an N2x6 NumPy array (X, Y, Z, R, G, B)
                                 that will be aligned to pcd1_data. Color values should be in the range [0, 1].

    Returns:
        np.ndarray: A combined (N1+N2)x6 NumPy array of the aligned point clouds.
        o3d.geometry.PointCloud: The combined and aligned Open3D point cloud object.
    """
    print("--- Starting Point Cloud Registration with Open3D ---")

    # 1. Convert NumPy arrays to Open3D PointCloud objects
    # ---------------------------------------------------------
    print("1. Converting NumPy arrays to Open3D PointCloud objects...")
    
    # Target Point Cloud (pcd1)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pcd1_data[:, :3])
    target.colors = o3d.utility.Vector3dVector(pcd1_data[:, 3:])
    
    # Source Point Cloud (pcd2)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcd2_data[:, :3])
    source.colors = o3d.utility.Vector3dVector(pcd2_data[:, 3:])

    # Create a copy of the source to transform
    source_to_transform = copy.deepcopy(source)

    # --- KEY PARAMETER TO TUNE ---
    # Voxel size for downsampling. This is the most critical parameter.
    # Try adjusting it based on the density of your point cloud.
    # A smaller value (e.g., 0.02, 0.01) can lead to more precision but requires more memory/time.
    voxel_size = 0.05 
    
    # 2. Coarse Alignment (Global Registration)
    # ---------------------------------------------------------
    print("\n2. Performing Coarse Alignment (Global Registration)...")

    # Downsample point clouds to speed up feature computation
    source_down = source_to_transform.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # Estimate normals for the downsampled clouds
    radius_normal = voxel_size * 2
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # Perform RANSAC-based registration on features
    distance_threshold = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    print(f"   Coarse alignment fitness: {ransac_result.fitness:.4f}")
    print(f"   Coarse alignment inlier_rmse: {ransac_result.inlier_rmse:.4f}")
    
    # Apply the coarse transformation to the full-resolution source cloud
    source_to_transform.transform(ransac_result.transformation)

    # 3. Fine Alignment (Local Registration) - IMPROVED
    # ---------------------------------------------------------
    print("\n3. Performing Fine Alignment (Point-to-Plane ICP)...")
    
    # Estimate normals for the original, full-resolution point clouds.
    # Point-to-Plane ICP requires the target cloud to have normals.
    print("   Estimating normals for full-resolution clouds...")
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    source_to_transform.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # --- KEY PARAMETER TO TUNE ---
    # This is the distance threshold for the fine-tuning ICP step.
    # Try making it smaller (e.g., voxel_size * 0.2) if the coarse alignment is good.
    icp_threshold = voxel_size * 0.4
    
    # **IMPROVEMENT**: Use Point-to-Plane ICP
    icp_result = o3d.pipelines.registration.registration_icp(
        source_to_transform, target, icp_threshold, np.identity(4), # Start from current alignment
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), # Using PointToPlane
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=200))

    print(f"   ICP fitness: {icp_result.fitness:.4f}")
    print(f"   ICP inlier_rmse: {icp_result.inlier_rmse:.4f}")

    # Apply the fine transformation
    source_to_transform.transform(icp_result.transformation)

    # 4. Combine and Post-process Point Clouds
    # ---------------------------------------------------------
    print("\n4. Combining and post-processing point clouds...")
    combined_pcd = target + source_to_transform

    # **NEW**: Optional post-processing steps to clean the final cloud

    # a) Statistical Outlier Removal
    # This can remove sparse outliers from the combined cloud.
    print("   Removing statistical outliers...")
    cl, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    combined_pcd = combined_pcd.select_by_index(ind)

    # b) Voxel Down-sampling
    # This reduces the number of points and can smooth out noisy areas.
    # Use a smaller voxel size here to preserve detail.
    final_voxel_size = voxel_size * 0.5
    print(f"   Down-sampling final cloud with voxel size: {final_voxel_size}")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=final_voxel_size)
    
    # 5. Convert Final Cloud to NumPy
    # ---------------------------------------------------------
    print("\n5. Converting final cloud to NumPy array...")
    combined_points = np.asarray(combined_pcd.points)
    combined_colors = np.asarray(combined_pcd.colors)
    
    # Concatenate to form the final N_combined x 6 array
    final_aligned_array = np.hstack((combined_points, combined_colors))

    print("\n--- Registration Complete ---")
    
    return final_aligned_array, combined_pcd

if __name__ == '__main__':
    # --- Example Usage ---
    # Create two synthetic point clouds that are slightly transformed versions of each other.
    
    print("--- Generating Synthetic Data for Demonstration ---")
    
    # Load a sample point cloud
    # try:
    #     sample_pcd = o3d.io.read_point_cloud(o3d.data.BunnyMesh().path)
    #     sample_pcd.scale(1 / np.max(sample_pcd.get_max_bound() - sample_pcd.get_min_bound()), center=sample_pcd.get_center())
    #     sample_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray color
    #     sample_pcd.estimate_normals()
    # except Exception as e:
    #     print(f"Could not load sample data. Error: {e}")
    #     # Create a simple box as a fallback
    #     sample_pcd = o3d.geometry.TriangleMesh.create_box().sample_points_poisson_disk(5000)
    #     sample_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    #     sample_pcd.estimate_normals()

    # pcd1: The target point cloud
    # pcd1 = copy.deepcopy(sample_pcd)
    # pcd1.paint_uniform_color([1, 0.7, 0]) # Paint it orange
    
    # # pcd2: The source point cloud, translated and rotated
    # pcd2 = copy.deepcopy(sample_pcd)
    # rotation = pcd2.get_rotation_matrix_from_xyz((0, np.pi / 4, 0)) # 45-degree rotation around Y-axis
    # pcd2.rotate(rotation, center=(0,0,0))
    # pcd2.translate((0.3, 0, 0.3)) # Translate it
    # pcd2.paint_uniform_color([0, 0.65, 0.93]) # Paint it blue

    # # Convert to NumPy arrays (as per the function's input requirement)
    # pcd1_data = np.hstack((np.asarray(pcd1.points), np.asarray(pcd1.colors)))
    # pcd2_data = np.hstack((np.asarray(pcd2.points), np.asarray(pcd2.colors)))
    data1 = np.load('tmp1.npz')
    pcd1_point = data1['raw_point_cloud'].reshape((-1,3))
    pcd1_color = data1['raw_point_color'].reshape((-1,3))
    pcd1_conf = data1['raw_point_conf'].reshape(-1)
    conf_threshold = np.percentile(pcd1_conf, 70)
    pcd1_mask = (pcd1_conf >= conf_threshold)
    pcd1_point = pcd1_point[pcd1_mask,:]
    pcd1_color = pcd1_color[pcd1_mask,:]
    pcd1_data = np.hstack((pcd1_point, pcd1_color))
    data2 = np.load('tmp2.npz')
    pcd2_point = data2['raw_point_cloud'].reshape((-1,3))
    pcd2_color = data2['raw_point_color'].reshape((-1,3))
    pcd2_conf = data2['raw_point_conf'].reshape(-1)
    conf_threshold = np.percentile(pcd2_conf, 70)
    pcd2_mask = (pcd2_conf >= conf_threshold)
    pcd2_point = pcd2_point[pcd2_mask,:]
    pcd2_color = pcd2_color[pcd2_mask,:]
    pcd2_data = np.hstack((pcd2_point, pcd2_color))

    # =================================================================================
    
    print(f"Shape of pcd1_data: {pcd1_data.shape}")
    print(f"Shape of pcd2_data: {pcd2_data.shape}")

    # Visualize the initial, unaligned point clouds
    # We need to convert them to Open3D objects just for visualization
    pcd1_vis = o3d.geometry.PointCloud()
    pcd1_vis.points = o3d.utility.Vector3dVector(pcd1_data[:,:3])
    pcd1_vis.colors = o3d.utility.Vector3dVector(pcd1_data[:,3:])
    
    pcd2_vis = o3d.geometry.PointCloud()
    pcd2_vis.points = o3d.utility.Vector3dVector(pcd2_data[:,:3])
    pcd2_vis.colors = o3d.utility.Vector3dVector(pcd2_data[:,3:])
    
    print("\nDisplaying initial unaligned point clouds. Close the window to continue.")
    o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis], window_name="Initial Unaligned Clouds")

    # Run the alignment function. It takes your NumPy arrays directly.
    aligned_data, aligned_pcd_obj = align_point_clouds_fast(pcd1_data, pcd2_data)

    print(f"\nShape of final aligned data array: {aligned_data.shape}")

    # Visualize the final, aligned point cloud
    print("\nDisplaying final aligned point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([aligned_pcd_obj], window_name="Final Aligned Cloud")
