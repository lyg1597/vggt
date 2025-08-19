import open3d as o3d
import numpy as np
import argparse
import copy
import sys

def align_point_clouds(source_pcd, target_pcd, voxel_size):
    """
    Aligns a source point cloud to a target point cloud using a coarse-to-fine approach.
    
    Args:
        source_pcd (o3d.geometry.PointCloud): The point cloud to be moved.
        target_pcd (o3d.geometry.PointCloud): The stationary target point cloud.
        voxel_size (float): The voxel size used for downsampling and feature estimation.
                            This is the most important parameter to tune.
                            
    Returns:
        o3d.geometry.PointCloud: The transformed source point cloud.
        np.ndarray: The final 4x4 transformation matrix.
    """
    print("--- Starting Coarse-to-Fine Alignment ---")
    
    # Create copies to avoid modifying the original point clouds
    source_transformed = copy.deepcopy(source_pcd)
    
    # --- 1. Coarse Alignment (Global Registration) ---
    print("\n1. Downsampling and extracting FPFH features...")
    source_down = source_transformed.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals for both downsampled clouds
    radius_normal = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    print("2. Running Fast Global Registration (FGR)...")
    distance_threshold = voxel_size * 1.5
    coarse_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
            
    print(f"   FGR Fitness: {coarse_result.fitness:.4f}, RMSE: {coarse_result.inlier_rmse:.4f}")

    # --- 2. Fine Alignment (Local Refinement) ---
    print("\n3. Running Point-to-Plane ICP for refinement...")
    
    # Estimate normals for the original target cloud (required for Point-to-Plane ICP)
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Use a smaller distance threshold for the fine-tuning ICP step
    icp_threshold = voxel_size * 0.4
    fine_result = o3d.pipelines.registration.registration_icp(
        source_transformed, target_pcd, icp_threshold, coarse_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
    print(f"   ICP Fitness: {fine_result.fitness:.4f}, RMSE: {fine_result.inlier_rmse:.4f}")
    
    # Apply the final transformation to the source point cloud
    final_transform = fine_result.transformation
    source_transformed.transform(final_transform)
    
    print("\n--- Alignment Complete ---")
    
    return source_transformed, final_transform

def main(args):
    """
    Main function to load, align, and save point clouds.
    """
    # 1. Load source and target point clouds
    print(f"Loading source cloud from: {args.source}")
    try:
        source_pcd = o3d.io.read_point_cloud(args.source)
        if not source_pcd.has_points():
            raise IOError
    except (IOError, RuntimeError):
        print(f"Error: Could not read source file at {args.source}")
        sys.exit(1)

    print(f"Loading target cloud from: {args.target}")
    try:
        target_pcd = o3d.io.read_point_cloud(args.target)
        if not target_pcd.has_points():
            raise IOError
    except (IOError, RuntimeError):
        print(f"Error: Could not read target file at {args.target}")
        sys.exit(1)

    # 2. Visualize initial state if requested
    if args.visualize:
        print("Displaying initial unaligned clouds (Source is Orange, Target is Blue)...")
        source_pcd_vis = copy.deepcopy(source_pcd)
        source_pcd_vis.paint_uniform_color([1, 0.7, 0]) # Orange
        target_pcd.paint_uniform_color([0, 0.65, 0.93]) # Blue
        o3d.visualization.draw_geometries([source_pcd_vis, target_pcd], window_name="Initial Alignment")

    # 3. Perform the alignment
    source_aligned, final_transform = align_point_clouds(source_pcd, target_pcd, args.voxel_size)

    # 4. Combine the aligned source and original target clouds
    combined_pcd = target_pcd + source_aligned

    # 5. Save the result
    print(f"\nSaving aligned and combined point cloud to: {args.output}")
    o3d.io.write_point_cloud(args.output, combined_pcd)
    print("Save complete.")

    # 6. Visualize final result if requested
    if args.visualize:
        print("Displaying final aligned cloud...")
        o3d.visualization.draw_geometries([combined_pcd], window_name="Final Aligned Result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align a source point cloud to a target point cloud using Open3D.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source", type=str, help="Path to the source .ply file (the one that will be moved).")
    parser.add_argument("target", type=str, help="Path to the target .ply file (the one that is stationary).")
    parser.add_argument(
        "--output", 
        type=str, 
        default="aligned_result.ply", 
        help="Path to save the combined and aligned .ply file."
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for down-sampling. This is the most important parameter to tune.\n"
             "It should be set based on the scale and density of your point clouds."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, display the point clouds before and after alignment."
    )
    
    args = parser.parse_args()

    # --- How to Run ---
    #
    # 1. Save this file as 'aligner.py'.
    # 2. Make sure you have Open3D installed (`pip install open3d`).
    #
    # Example command:
    # python aligner.py path/to/source.ply path/to/target.ply --output final.ply --visualize --voxel_size 0.02
    #
    # ------------------

    main(args)
