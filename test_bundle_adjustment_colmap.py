import os
import numpy as np
import tempfile
import shutil
import pycolmap

def qvec2rotmat(qvec):
    """
    Convert a quaternion to a rotation matrix.
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[3] * qvec[0],
         2 * qvec[1] * qvec[3] + 2 * qvec[2] * qvec[0]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[3] * qvec[0],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[1] * qvec[0]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[2] * qvec[0],
         2 * qvec[2] * qvec[3] + 2 * qvec[1] * qvec[0],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def rotmat2qvec(R):
    """
    Convert a rotation matrix to a quaternion.
    """
    q = np.zeros(4)
    t = np.trace(R)
    if t > 0:
        t = np.sqrt(t + 1)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return q


def bundle_adjustment(extrinsics, all_points_3d, all_observations, intrinsics):
    """
    Performs bundle adjustment using pycolmap.

    Args:
        extrinsics (np.ndarray): N_images x 4 x 4 array of camera extrinsics.
        all_points_3d (np.ndarray): N_points x 3 array of 3D points.
        all_observations (list): A list of lists of observations for each image.
                                 Each observation is a tuple (point3D_id, x, y).
        intrinsics (np.ndarray): A 3x3 camera intrinsics matrix.

    Returns:
        tuple: A tuple containing:
            - optimized_poses (np.ndarray): Optimized N_images x 4 x 4 camera extrinsics.
            - optimized_points (np.ndarray): Optimized N_points x 3 array of 3D points.
    """
    # Create a reconstruction object
    recon = pycolmap.Reconstruction()

    # Add camera
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    recon.add_camera(pycolmap.Camera(
        model='SIMPLE_PINHOLE',
        width=1,  # Dummy values
        height=1, # Dummy values
        params=[fx, cx, cy]
    ))

    # Add images and 2D points
    for i, (ext, obs) in enumerate(zip(extrinsics, all_observations)):
        R = ext[:3, :3]
        t = ext[:3, 3]
        qvec = rotmat2qvec(R)
        
        image = pycolmap.Image(
            # id=i,
            qvec=qvec,
            tvec=t,
            camera_id=0,
            name=f"image{i}.jpg"
        )
        
        points2D = np.array([o[1:] for o in obs])
        point3D_ids = np.array([o[0] for o in obs], dtype=np.int64)
        
        image.points2D = pycolmap.ListPoint2D(points2D)
        for idx, p3d_id in enumerate(point3D_ids):
            image.points2D[idx].point3D_id = p3d_id
            
        recon.add_image(image)


    # Add 3D points
    for i, point in enumerate(all_points_3d):
        p3d = pycolmap.Point3D(
            id=i,
            xyz=point,
            color=np.array([255, 255, 255]),
            error=0.0
        )
        recon.add_point3D(p3d)
    
    # Run bundle adjustment
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_principal_point = False
    summary = pycolmap.bundle_adjuster(options, recon)

    if not summary.success:
        print("Error: Bundle adjustment failed.")
        return None, None

    print("Bundle adjustment completed successfully.")
    print(f"Initial cost: {summary.initial_cost}, Final cost: {summary.final_cost}")

    # Extract optimized data
    optimized_poses = []
    for image in recon.images.values():
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        optimized_poses.append(pose)
        
    optimized_points = np.array([p.xyz for p in recon.points3D.values()])

    return np.array(optimized_poses), optimized_points


if __name__ == '__main__':
    # --- Example Usage ---
    # This is a dummy example. Replace with your actual data.
    
    # Number of cameras and points
    num_cameras = 5
    num_points = 10

    # Dummy Intrinsics (fx, fy, cx, cy)
    intrinsics = np.array([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ])

    # Dummy initial 3D points
    all_points_3d = np.random.rand(num_points, 3) * 10

    # Dummy initial camera extrinsics (poses)
    extrinsics = []
    for i in range(num_cameras):
        pose = np.eye(4)
        pose[0, 3] = i * 0.5 # Slightly offset cameras
        extrinsics.append(pose)
    extrinsics = np.array(extrinsics)

    # Dummy observations
    # Each camera sees all points with some noise
    all_observations = []
    for i in range(num_cameras):
        observations_for_image = []
        for j in range(num_points):
            # Project 3D point to 2D
            point_cam_frame = np.linalg.inv(extrinsics[i]) @ np.append(all_points_3d[j], 1)
            point_2d_homogeneous = intrinsics @ point_cam_frame[:3]
            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            
            # Add some noise
            point_2d += np.random.randn(2) * 0.5
            
            observations_for_image.append((j, point_2d[0], point_2d[1]))
        all_observations.append(observations_for_image)

    # --- Run Bundle Adjustment ---
    optimized_poses, optimized_points = bundle_adjustment(
        extrinsics, all_points_3d, all_observations, intrinsics
    )

    if optimized_poses is not None and optimized_points is not None:
        print("\n--- Optimized Poses (first camera) ---")
        print(optimized_poses[0])
        print("\n--- Optimized 3D Points (first 5) ---")
        print(optimized_points[:5])
