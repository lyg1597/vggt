import open3d as o3d
import numpy as np
import argparse
import copy
import os

def load_pcd_from_npz(file_path):
    """
    Loads a point cloud from an .npz file.

    The .npz file must contain 'point_cloud' and 'point_cloud_colors' keys.

    Args:
        file_path (str): The path to the .npz file.

    Returns:
        o3d.geometry.PointCloud: An Open3D point cloud object, or None if loading fails.
    """
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    # Check for required keys in the .npz file
    if 'point_cloud' not in data or 'point_cloud_colors' not in data:
        print(f"Error: NPZ file {file_path} must contain 'point_cloud' and 'point_cloud_colors' arrays.")
        return None

    points = data['point_cloud']
    colors = data['point_cloud_colors']

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normalize colors to [0, 1] float range if they are in [0, 255] uint8 range
    if colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def align_point_clouds(source, target, use_global_alignment=True):
    """
    Aligns a source point cloud to a target point cloud.

    Performs a two-stage registration (FGR + Colored ICP) or a single-stage
    refinement (Colored ICP only), controlled by the 'use_global_alignment' flag.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud to align.
        target (o3d.geometry.PointCloud): The target point cloud.
        use_global_alignment (bool): If True, runs FGR for initial alignment.

    Returns:
        np.ndarray: The final 4x4 transformation matrix.
        o3d.geometry.PointCloud: A new point cloud object representing the transformed source.
    """
    if not source.has_colors() or not target.has_colors():
        raise ValueError("Both source and target point clouds must have colors.")

    if use_global_alignment:
        print("--- Starting 2-Stage Alignment (FGR + Colored ICP) ---")
        # STAGE 1: Fast Global Registration
        print("\n--- STAGE 1: Performing Fast Global Registration ---")
        voxel_size = 0.05

        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        radius_normal_fgr = voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_fgr, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_fgr, max_nn=30))

        radius_feature = voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        distance_threshold_fgr = voxel_size * 1.5
        fgr_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold_fgr))
        
        print(f"  > FGR Fitness: {fgr_result.fitness:.4f}")
        initial_transformation = fgr_result.transformation
    else:
        print("--- Starting 1-Stage Alignment (Colored ICP Only) ---")
        print("\n--- STAGE 1: SKIPPED (Global Alignment Disabled) ---")
        initial_transformation = np.identity(4)

    # STAGE 2: Colored ICP
    print("\n--- STAGE 2: Refining with Colored ICP ---")
    voxel_radius_icp = 0.02
    
    radius_normal_icp = voxel_radius_icp * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_icp, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_icp, max_nn=30))

    icp_result = o3d.pipelines.registration.registration_colored_icp(
        source, target, voxel_radius_icp, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    )

    print(f"  > Colored ICP Fitness: {icp_result.fitness:.4f}")
    
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(icp_result.transformation)

    print("\n--- Alignment Complete ---")
    return icp_result.transformation, source_transformed

def visualize_registration(source, target, transformed_source, window_title="Final Alignment"):
    """ Visualizes the point clouds before and after registration. """
    print("\nVisualizing initial state (Source & Target). Close the window to see the result.")
    o3d.visualization.draw_geometries([source, target], window_name="Initial State")

    print(f"Visualizing final state (Transformed Source in green).")
    transformed_source.paint_uniform_color([0.1, 0.9, 0.1])
    o3d.visualization.draw_geometries([transformed_source, target], window_name=window_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align two colored point clouds from .npz files and visualize the result.")
    parser.add_argument("source", type=str, help="Path to the source .npz file.")
    parser.add_argument("target", type=str, help="Path to the target .npz file.")
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Skip the initial global alignment (FGR) and run only Colored ICP."
    )
    args = parser.parse_args()

    # --- 1. Load Point Clouds from NPZ ---
    print(f"Loading source cloud from: {args.source}")
    source_pcd = load_pcd_from_npz(args.source)
    if source_pcd is None:
        exit()

    print(f"Loading target cloud from: {args.target}")
    target_pcd = load_pcd_from_npz(args.target)
    if target_pcd is None:
        exit()

    if not source_pcd.has_points() or not target_pcd.has_points():
        print("Error: One or both point clouds are empty after loading.")
        exit()
        
    # --- 2. Align Point Clouds ---
    try:
        use_global = not args.no_global
        
        # visualize_registration(source_pcd, target_pcd, source_pcd, window_title="Before")


        final_transform, source_pcd_transformed = align_point_clouds(source_pcd, target_pcd, use_global_alignment=use_global)
        
        print("\nFinal Transformation Matrix:")
        print(final_transform)
        
        # --- 3. Visualize ---
        title = "Final Alignment (FGR + ICP)" if use_global else "Final Alignment (ICP Only)"
        visualize_registration(source_pcd, target_pcd, source_pcd_transformed, window_title=title)

    except ValueError as e:
        print(f"\nError during alignment: {e}")