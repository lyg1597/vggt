import g2o
import numpy as np

def bundle_adjustment(poses, points, observations, intrinsics, iterations=10, fix_first_pose=True):
    """
    Performs bundle adjustment using g2o.

    Args:
        poses (list): A list of initial g2o.SE3Quat camera poses.
        points (list): A list of initial 3D np.array points.
        observations (list): A list of tuples, where each tuple is
                             (point_id, pose_id, measurement_uv).
        intrinsics (np.array): Camera intrinsics [fx, fy, cx, cy].
        iterations (int): Number of optimization iterations.
        fix_first_pose (bool): If True, the first camera pose is held constant.

    Returns:
        tuple: A tuple containing (optimized_poses, optimized_points).
    """
    # 1. Setup the optimizer
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3ProjectXYZ()
    solver = g2o.LinearSolverCSparse(solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    # 2. Add camera intrinsics as a parameter
    focal_length = intrinsics[0]
    principal_point = (intrinsics[2], intrinsics[3])
    cam_params = g2o.CameraParameters(focal_length, principal_point, 0)
    cam_params.set_id(0)
    optimizer.add_parameter(cam_params)

    # 3. Add vertices (camera poses and 3D points)
    # Add pose vertices
    for i, pose in enumerate(poses):
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        v_se3.set_estimate(pose)
        if i == 0 and fix_first_pose:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)

    # Add point vertices
    point_id_offset = len(poses)
    for i, point in enumerate(points):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(i + point_id_offset)
        v_p.set_estimate(point)
        v_p.set_marginalized(True)
        optimizer.add_vertex(v_p)

    # 4. Add edges (2D observations)
    for point_id, pose_id, measurement in observations:
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, optimizer.vertex(point_id + point_id_offset))
        edge.set_vertex(1, optimizer.vertex(pose_id))
        edge.set_measurement(measurement)
        edge.set_information(np.identity(2)) # Assume 1-pixel error
        edge.set_parameter_id(0, 0) # Use camera parameters with id 0
        optimizer.add_edge(edge)

    # 5. Run the optimization
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(iterations)

    # 6. Extract optimized results
    optimized_poses = []
    for i in range(len(poses)):
        optimized_poses.append(optimizer.vertex(i).estimate())
        
    optimized_points = []
    for i in range(len(points)):
        optimized_points.append(optimizer.vertex(i + point_id_offset).estimate())
        
    return optimized_poses, optimized_points

def pose_matrix_to_se3(pose_matrix):
    """Converts a 4x4 numpy pose matrix to a g2o.SE3Quat object."""
    rotation_matrix = pose_matrix[:3, :3]
    translation_vector = pose_matrix[:3, 3]
    return g2o.SE3Quat(rotation_matrix, translation_vector)