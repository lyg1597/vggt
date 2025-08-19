import torch
import cv2
import numpy as np
import g2o
import argparse
from pathlib import Path
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# --- SuperGlue Imports ---
# Make sure the SuperGluePretrainedNetwork directory is accessible
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor

def triangulate_points(pts0, pts1, pose0, pose1, K):
    """
    Triangulates 3D points from two sets of 2D correspondences.

    Args:
        pts0 (np.array): 2D points in the first image (N, 2).
        pts1 (np.array): 2D points in the second image (N, 2).
        pose0 (np.array): 4x4 extrinsic matrix for the first camera.
        pose1 (np.array): 4x4 extrinsic matrix for the second camera.
        K (np.array): 3x3 intrinsic matrix.

    Returns:
        np.array: Triangulated 3D points in the world frame (N, 3).
    """
    # Projection matrices
    P0 = K @ np.linalg.inv(pose0)[:3]
    P1 = K @ np.linalg.inv(pose1)[:3]

    # Triangulate points using OpenCV
    points_4d_hom = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)

    # Convert from homogeneous to 3D coordinates
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
    return points_3d

def bundle_adjustment_scipy(poses, points, observations, K, iterations=100):
    """
    Performs bundle adjustment using SciPy's least_squares.
    This is a more stable alternative to g2o.
    """
    
    def project(points, pose, K):
        """Projects 3D points to 2D image plane."""
        # Convert pose (rotation vector, translation vector) to projection matrix
        R_mat, _ = cv2.Rodrigues(pose[:3])
        t_vec = pose[3:]
        
        # Project points
        points_cam = R_mat @ points.T + t_vec[:, np.newaxis]
        points_proj = K @ points_cam
        
        # Normalize
        points_2d = points_proj[:2, :] / points_proj[2, :]
        return points_2d.T

    def residuals(params, n_poses, n_points, K, observations):
        """
        Compute the residuals (reprojection errors) for the optimizer.
        """
        # 1. Unpack parameters
        pose_params = params[:n_poses * 6].reshape((n_poses, 6))
        point_params = params[n_poses * 6:].reshape((n_points, 3))
        
        # 2. Compute residuals for all observations
        all_residuals = []
        for point_id, pose_id, measurement in observations:
            # Get the corresponding 3D point and pose
            point_3d = point_params[point_id]
            pose = pose_params[pose_id]
            
            # Project the point and compute the error
            projected_point = project(point_3d[np.newaxis, :], pose, K)[0]
            error = projected_point - measurement
            all_residuals.append(error)
            
        return np.concatenate(all_residuals)

    # --- 1. Pack initial parameters into a single vector ---
    # Convert rotation matrices to rotation vectors (3 params) + translation (3 params)
    initial_pose_params = []
    for pose_mat in poses:
        r_vec, _ = cv2.Rodrigues(pose_mat[:3, :3])
        t_vec = pose_mat[:3, 3]
        initial_pose_params.append(np.concatenate((r_vec.flatten(), t_vec)))
    
    # Flatten everything into a single 1D array
    initial_params = np.concatenate((np.array(initial_pose_params).flatten(), np.array(points).flatten()))
    
    # --- 2. Run the optimization ---
    print("\n--- Starting Bundle Adjustment with SciPy ---")
    n_poses = len(poses)
    n_points = len(points)
    
    # The first pose is fixed by not including its parameters in the optimization
    # Here we handle it by setting its parameters to be constant.
    # A more robust way is to re-parameterize, but for simplicity, we optimize all.
    # The global frame is implicitly defined by the initial poses.
    
    res = least_squares(
        residuals,
        initial_params,
        args=(n_poses, n_points, K, observations),
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        max_nfev=iterations
    )

    # --- 3. Unpack the optimized parameters ---
    optimized_params = res.x
    opt_pose_params = optimized_params[:n_poses * 6].reshape((n_poses, 6))
    optimized_points = optimized_params[n_poses * 6:].reshape((n_points, 3))

    optimized_poses = []
    for pose_param in opt_pose_params:
        r_vec = pose_param[:3]
        t_vec = pose_param[3:]
        R_mat, _ = cv2.Rodrigues(r_vec)
        
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = R_mat
        pose_mat[:3, 3] = t_vec
        optimized_poses.append(pose_mat)

    print("--- Optimization Finished ---")
    return np.array(optimized_poses), np.array(optimized_points)


def visualize_reconstruction_open3d(poses, points):
    """
    Visualizes the camera poses and 3D point cloud using Open3D.

    Args:
        poses (np.array): Array of 4x4 camera pose matrices.
        points (np.array): Array of 3D points (N, 3).
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.0, 1.0]) # Blue points

    # Create a list of geometries to draw
    geometries = [pcd]

    # Create coordinate frames for each camera pose
    for pose in poses:
        # Create a coordinate frame mesh
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # Transform the frame to the camera's pose
        frame.transform(pose)
        geometries.append(frame)

    # Visualize the geometries
    print("\nVisualizing with Open3D. Press 'q' to close the window.")
    o3d.visualization.draw_geometries(geometries)

def dense_reconstruction(images, poses, K):
    """
    Performs dense 3D reconstruction from a sequence of images and refined poses.

    Args:
        images (np.array): Array of N images (N, H, W, 3) in RGB format.
        poses (np.array): Array of N refined 4x4 extrinsic matrices.
        K (np.array): Single 3x3 intrinsic matrix.

    Returns:
        o3d.geometry.PointCloud: The combined dense point cloud.
    """
    print("\n--- Starting Dense Reconstruction ---")
    all_points = []
    all_colors = []
    
    # Use StereoSGBM for better results
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=128,
                                   blockSize=5,
                                   P1=8 * 3 * 5**2,
                                   P2=32 * 3 * 5**2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    for i in range(len(images) - 1):
        print(f"  - Processing pair {i} and {i+1}")
        
        # Get poses and images for the pair
        pose0, pose1 = poses[i], poses[i+1]
        img0_rgb, img1_rgb = images[i], images[i+1]
        
        # Convert to grayscale for stereo matching
        img0_gray = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2GRAY)
        img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        
        # --- Stereo Rectification ---
        # Calculate relative transformation from camera 0 to camera 1
        T_0_to_1 = pose1 @ np.linalg.inv(pose0)
        R_rel = T_0_to_1[:3, :3]
        t_rel = T_0_to_1[:3, 3]
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, None, K, None, img0_gray.shape[::-1], R_rel, t_rel
        )
        
        map1_x, map1_y = cv2.initUndistortRectifyMap(K, None, R1, P1, img0_gray.shape[::-1], cv2.CV_32FC1)
        map2_x, map2_y = cv2.initUndistortRectifyMap(K, None, R2, P2, img1_gray.shape[::-1], cv2.CV_32FC1)

        img0_rect = cv2.remap(img0_gray, map1_x, map1_y, cv2.INTER_LINEAR)
        img1_rect = cv2.remap(img1_gray, map2_x, map2_y, cv2.INTER_LINEAR)
        
        # --- Disparity Calculation ---
        disparity = stereo.compute(img0_rect, img1_rect).astype(np.float32) / 16.0
        
        # --- 3D Point Cloud Generation ---
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Get colors from the original image (before rectification)
        colors = cv2.cvtColor(img0_rgb, cv2.COLOR_RGB2BGR)
        
        # Filter out invalid points
        mask = disparity > disparity.min()
        points_3d = points_3d[mask]
        colors = colors[mask]
        
        # --- Transform points from camera frame to world frame ---
        # The reprojectImageTo3D gives points relative to the *rectified* camera 0 frame.
        # We need to transform them back to the original camera 0 frame, then to world.
        # 1. Get the inverse rectification rotation
        R1_inv = np.linalg.inv(R1)
        # 2. Transform points from rectified camera 0 to original camera 0 frame
        points_in_cam0_frame = (R1_inv @ points_3d.T).T
        # 3. Transform points from camera 0 frame to world frame
        pose0_inv = np.linalg.inv(pose0)
        points_3d_world = (pose0_inv[:3, :3] @ points_in_cam0_frame.T + pose0_inv[:3, 3, np.newaxis]).T
        
        all_points.append(points_3d_world)
        all_colors.append(colors)

    # Combine all point clouds
    final_points = np.concatenate(all_points, axis=0)
    final_colors = np.concatenate(all_colors, axis=0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors / 255.0) # Normalize colors to [0, 1]
    
    # Downsample to manage memory
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    
    return pcd


def reconstruction_pipeline(images, extrinsics, intrinsics, device='cpu', confidence_thresh=0.5):
    """
    Full 3D reconstruction pipeline using SuperGlue and g2o Bundle Adjustment.

    Args:
        images (np.array): Array of N images (N, H, W, 3) in RGB format.
        extrinsics (np.array): Array of N 4x4 extrinsic matrices (N, 4, 4).
        intrinsics (np.array): Array of N 3x3 intrinsic matrices (N, 3, 3).
        device (str): 'cpu' or 'cuda'.
        confidence_thresh (float): SuperGlue match confidence threshold.

    Returns:
        tuple: (optimized_poses, optimized_points)
    """
    # --- 1. SuperGlue Feature Matching and Initial Triangulation ---
    print("--- Step 1: Feature Matching & Initial Triangulation ---")
    
    # Initialize SuperGlue model
    superglue_config = {
        'superpoint': {'max_keypoints': 2048},
        'superglue': {'weights': 'outdoor', 'match_threshold': 0.2}
    }
    matching = Matching(superglue_config).eval().to(device)
    
    # Data structures for BA
    all_points_3d = []
    all_observations = []
    point_map = {}  # Tracks which 3D point a 2D feature corresponds to
    point_counter = 0

    for i in range(len(images) - 1):
        print(f"  - Matching frame {i} and {i+1}")
        
        # Convert RGB images to BGR for OpenCV processing
        img0_bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        img1_bgr = cv2.cvtColor(images[i+1], cv2.COLOR_RGB2BGR)
        
        pose0, pose1 = extrinsics[i], extrinsics[i+1]
        K = intrinsics[i] # Assume intrinsics are similar

        # Preprocess images for SuperGlue (expects grayscale)
        img0_gray = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        tensor0 = frame2tensor(img0_gray, device)
        tensor1 = frame2tensor(img1_gray, device)

        # Perform matching
        with torch.no_grad():
            pred = matching({'image0': tensor0, 'image1': tensor1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        
        # Filter matches by confidence
        valid = (pred['matches0'] > -1) & (pred['matching_scores0'] > confidence_thresh)
        matches_idx = pred['matches0'][valid]
        
        kpts0 = pred['keypoints0'][valid]
        kpts1 = pred['keypoints1'][matches_idx]
        
        if len(kpts0) < 8: # Need at least 8 points for triangulation
            continue

        # Triangulate initial 3D points
        points_3d_local = triangulate_points(kpts0, kpts1, pose0, pose1, K)

        # Add points and observations for Bundle Adjustment
        for j in range(len(kpts0)):
            # Check if the point from frame i has been seen before
            # A unique key for a 2D point is its (frame_id, keypoint_index)
            # Here we use the point coordinates as a proxy for index
            pt0_key = (i, tuple(kpts0[j]))
            
            if pt0_key in point_map:
                # This 2D point is already part of an existing 3D track
                point_3d_id = point_map[pt0_key]
            else:
                # This is a new 3D point
                point_3d_id = point_counter
                all_points_3d.append(points_3d_local[j])
                # Add the first observation of this new point
                all_observations.append((point_3d_id, i, kpts0[j]))
                point_counter += 1

            # Link the 2D point in the next frame to this 3D track
            pt1_key = (i + 1, tuple(kpts1[j]))
            point_map[pt1_key] = point_3d_id
            # Add the second observation
            all_observations.append((point_3d_id, i + 1, kpts1[j]))

    print(f"Generated {len(all_points_3d)} initial 3D points and {len(all_observations)} observations.")

    # --- 2. Bundle Adjustment ---
    print("\n--- Step 2: Refining with Bundle Adjustment ---")
    # Convert extrinsics array to a list of poses for g2o functionbundle_adjustment
    optimized_poses, optimized_points = bundle_adjustment_scipy(
        list(extrinsics), all_points_3d, all_observations, intrinsics[0]
    )
    
    dense_point_cloud = dense_reconstruction(images, optimized_poses, intrinsics[0])
    
    return optimized_poses, dense_point_cloud



if __name__ == '__main__':
    # --- This block simulates the inputs you would provide ---
    
    # Check if SuperGlue repo exists
    if not Path('SuperGluePretrainedNetwork').exists():
        print("Error: 'SuperGluePretrainedNetwork' directory not found.")
        print("Please clone the repository: git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git")
        exit()

    # 1. SIMULATE INPUT DATA
    print("--- Read input data (replace with your actual data) ---")
    datapack = np.load(f'step_res/res_raw_00000.npz')
    extrinsics_tmp = datapack['extrinsics']
    extrinsics_array = np.zeros((extrinsics_tmp.shape[0], 4,4))
    extrinsics_array[:,3,3] = 1
    extrinsics_array[:,:3,:] = extrinsics_tmp
    intrinsics_array = datapack['intrinsics']
    images_rgb_array = datapack['images']
    
    # # Intrinsics
    # K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    
    # # Images (dummy images with moving objects)
    # def create_test_image(text, translation_x):
    #     # This creates a BGR image using OpenCV
    #     img = np.full((480, 640, 3), 20, dtype=np.uint8)
    #     cv2.rectangle(img, (100 + translation_x, 100), (250 + translation_x, 350), (100, 255, 100), 5)
    #     cv2.circle(img, (400 + translation_x, 220), 80, (255, 150, 100), -1)
    #     cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
    #     return img

    # # Create a list of BGR images
    # bgr_images_list = [create_test_image('1', 0), create_test_image('2', 40), create_test_image('3', 90)]
    
    # # Convert to a single (N, H, W, 3) RGB array for the pipeline
    # images_rgb_array = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in bgr_images_list])
    
    # # Extrinsics (camera poses as 4x4 matrices)
    # extrinsics_list = [
    #     np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=float),
    #     np.array([[1, 0, 0, -0.5], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=float),
    #     np.array([[1, 0, 0, -1.0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]], dtype=float)
    # ]
    # extrinsics_array = np.array(extrinsics_list)

    # # Intrinsics
    # intrinsics_list = [K for _ in images_rgb_array]
    # intrinsics_array = np.array(intrinsics_list)

    # 2. RUN THE PIPELINE
    final_poses, final_points = reconstruction_pipeline(
        images_rgb_array, extrinsics_array, intrinsics_array, device='cuda'
    )

    # 3. OUTPUT RESULTS
    print("\n--- Step 3: Final Results ---")
    print(f"Refined {final_poses.shape[0]} camera poses.")
    print(f"Reconstructed {final_points.shape[0]} 3D points.")
    
    visualize_reconstruction_open3d(final_poses, final_points)

    # You can now save `final_poses` and `final_points` to a file,
    # or visualize them using a library like Open3D or Matplotlib.
    print("\nPipeline finished.")

