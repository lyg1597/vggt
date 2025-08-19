import open3d as o3d
import numpy as np
import argparse
import copy
import os

def align_point_clouds(source, target, use_global_alignment=True):
    """
    Aligns a source point cloud to a target point cloud.

    Can perform a two-stage registration (FGR + Colored ICP) or a single-stage
    refinement (Colored ICP only), controlled by the 'use_global_alignment' flag.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud to align.
        target (o3d.geometry.PointCloud): The target point cloud.
        use_global_alignment (bool): If True, runs FGR for initial alignment.
                                     If False, skips FGR and starts ICP from an
                                     identity transformation.

    Returns:
        np.ndarray: The final 4x4 transformation matrix.
        o3d.geometry.PointCloud: A new point cloud object representing the transformed source.
    """
    if not source.has_colors() or not target.has_colors():
        raise ValueError("Both source and target point clouds must have colors.")

    # --- Set initial transformation ---
    if use_global_alignment:
        print("--- Starting 2-Stage Alignment (FGR + Colored ICP) ---")
        # --- STAGE 1: Fast Global Registration (Geometric Alignment) ---
        print("\n--- STAGE 1: Performing Fast Global Registration ---")
        voxel_size = 0.05  # Voxel size for FGR

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
        # Start with an identity matrix if global alignment is skipped
        initial_transformation = np.identity(4)

    # --- STAGE 2: Colored ICP (Fine-tuning) ---
    print("\n--- STAGE 2: Refining with Colored ICP ---")
    voxel_radius_icp = 0.02

    # Normals are required for the geometric term in Colored ICP
    radius_normal_icp = voxel_radius_icp * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_icp, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_icp, max_nn=30))

    icp_result = o3d.pipelines.registration.registration_colored_icp(
        source, target, voxel_radius_icp, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    )

    print(f"  > Colored ICP Fitness: {icp_result.fitness:.4f}")
    
    # Create a transformed copy of the original source cloud
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(icp_result.transformation)

    print("\n--- Alignment Complete ---")
    return icp_result.transformation, source_transformed

def visualize_registration(source, target, transformed_source, window_title="Final Alignment"):
    """ Visualizes the point clouds before and after registration. """
    print("\nVisualizing initial state (Source & Target). Close the window to see the result.")
    o3d.visualization.draw_geometries([source, target], window_name="Initial State")

    print(f"Visualizing final state (Transformed Source in green).")
    transformed_source.paint_uniform_color([0.1, 0.9, 0.1]) # Bright green
    o3d.visualization.draw_geometries([transformed_source, target], window_name=window_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align two colored point clouds and visualize the result.")
    parser.add_argument("source", type=str, help="Path to the source point cloud file (e.g., .ply, .pcd).")
    parser.add_argument("target", type=str, help="Path to the target point cloud file (e.g., .ply, .pcd).")
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Skip the initial global alignment (FGR) and run only Colored ICP. Use this if clouds are already roughly aligned."
    )
    args = parser.parse_args()

    # --- 1. Load Point Clouds ---
    print(f"Loading source cloud: {args.source}")
    if not os.path.exists(args.source):
        print(f"Error: Source file not found.")
        exit()
    source_pcd = o3d.io.read_point_cloud(args.source)

    print(f"Loading target cloud: {args.target}")
    if not os.path.exists(args.target):
        print(f"Error: Target file not found.")
        exit()
    target_pcd = o3d.io.read_point_cloud(args.target)

    if not source_pcd.has_points() or not target_pcd.has_points():
        print("Error: One or both point clouds are empty.")
        exit()
        
    # --- 2. Align Point Clouds ---
    try:
        # Determine whether to use global alignment based on the flag
        use_global = not args.no_global
        
        final_transform, source_pcd_transformed = align_point_clouds(source_pcd, target_pcd, use_global_alignment=use_global)
        
        print("\nFinal Transformation Matrix:")
        print(final_transform)
        
        # --- 3. Visualize ---
        title = "Final Alignment (FGR + ICP)" if use_global else "Final Alignment (ICP Only)"
        visualize_registration(source_pcd, target_pcd, source_pcd_transformed, window_title=title)

    except ValueError as e:
        print(f"\nError during alignment: {e}")