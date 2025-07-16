import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import json 
import pyvista as pv 
import trimesh

def solve_T_s(a: np.ndarray, b: np.ndarray,
              init_s: float | None = None) -> tuple[np.ndarray, float]:
    """
    Estimate T (4×4) and scalar s such that T @ (a with scaled translation) ≈ b.

    Parameters
    ----------
    a, b : ndarray, shape (N, 4, 4)
        Homogeneous transforms for two coordinate systems.
    init_s : float, optional
        Initial guess for the scale (defaults to 1.0 or the ratio of median
        translation norms).

    Returns
    -------
    T_hat : ndarray, shape (4, 4)
        Estimated similarity transform.
    s_hat : float
        Estimated translation scale factor.
    """
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.shape[1:] == (4, 4)
    N = a.shape[0]

    # Extract rotations and translations
    Ra = a[:, :3, :3]
    ta = a[:, :3, 3]
    Rb = b[:, :3, :3]
    tb = b[:, :3, 3]

    # ---------- initial guesses ------------------------------------------------
    # Rotation: average of R_b R_aᵀ (projected back to SO(3))
    R_init = Rotation.from_matrix(Rb @ np.transpose(Ra, (0, 2, 1))).mean()
    r_init = R_init.as_rotvec()
    # Translation: rough by aligning centroids (with s ≈ 1)
    if init_s is None:
        init_s = np.median(np.linalg.norm(tb, axis=1) /
                           np.maximum(np.linalg.norm(ta, axis=1), 1e-9))
    t_init = tb.mean(axis=0) - R_init.apply(init_s * ta.mean(axis=0))

    # Parameter vector p = [rx, ry, rz, tx, ty, tz, s]
    p0 = np.hstack([r_init, t_init, init_s])

    # ---------- residual --------------------------------------------------------
    def residual(p):
        r_vec, t_vec, s = p[:3], p[3:6], p[6]
        R = Rotation.from_rotvec(r_vec).as_matrix()
        # orientation residual: flatten matrices
        R_pred = R @ Ra                        # (N, 3, 3)
        orient_res = (R_pred - Rb).reshape(N, -1)
        # translation residual
        trans_pred = t_vec + (R @ (ta.T * s)).T
        trans_res = trans_pred - tb
        return np.concatenate([orient_res, trans_res], axis=1).ravel()

    # ---------- solve -----------------------------------------------------------
    res = least_squares(residual, p0, method="lm")   # Levenberg–Marquardt

    r_opt, t_opt, s_opt = res.x[:3], res.x[3:6], res.x[6]
    R_opt = Rotation.from_rotvec(r_opt).as_matrix()

    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    T_opt[:3, 3] = t_opt
    return T_opt, s_opt

# --- assume you already have -----------------------------------------------
# a, b               # (N, 4, 4) arrays of homogeneous transforms
# T_hat, s_hat       # returned by solve_T_s(a, b)
# ---------------------------------------------------------------------------

def a_to_b(a, T, s):
    """
    Apply T and scale s to take poses in frame A → frame B.

    b_pred_i = T @ [ R_a_i | s * t_a_i ]
    """
    a_scaled = a.copy()
    a_scaled[:, :3, 3] *= s               # scale the translation only
    return (T @ a_scaled)                 # broadcasted matrix product

def b_to_a(b, T, s):
    """
    Inverse mapping: recover A-frame poses from B-frame ones.

    a_pred_i = inv(T) @ b_i;  then un-scale the translation.
    """
    T_inv = np.linalg.inv(T)
    a_pred = T_inv @ b
    a_pred[:, :3, 3] /= s                 # undo the scale on translation
    return a_pred

def transform_point_cloud(points, T, s):
    """
    Transforms a point cloud from frame A (vggt) to frame B (json).

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) for point coordinates.
        T (np.ndarray): The 4x4 transformation matrix from solve_T_s.
        s (float): The scale factor from solve_T_s.

    Returns:
        np.ndarray: The transformed point cloud of shape (N, 3).
    """
    # First, scale the points
    points_scaled = points * s

    # To apply the 4x4 transform, we need to convert points to homogeneous coordinates (N, 4)
    # by adding a '1' at the end of each point vector.
    points_homogeneous = np.hstack([points_scaled, np.ones((points.shape[0], 1))])

    # Apply the transformation matrix T
    # T is (4, 4), points_homogeneous.T is (4, N). The result is (4, N).
    transformed_points_homogeneous = (T @ points_homogeneous.T).T

    # Convert back from homogeneous coordinates by dropping the last column
    return transformed_points_homogeneous[:, :3]

def create_point_cloud_animation(
    points: np.ndarray,
    colors: np.ndarray,
    camera_poses: np.ndarray,
    confidence: np.ndarray,
    conf_thresh: float,
    output_filename: str = "point_cloud_animation.gif",
    camera_frustum_size: float = 0.1,
):
    """
    Generates an animation of an incrementally built point cloud with camera visualization.

    Args:
        points (np.ndarray): Point cloud data in the form N*W*H*3.
        colors (np.ndarray): Color of each point in the form N*W*H*3.
        camera_poses (np.ndarray): N*4*4 camera transformation matrices.
        confidence (np.ndarray): Confidence value for each point in the form N*W*H*1.
        conf_thresh (float): The threshold for confidence filtering.
        output_filename (str, optional): The name of the output animation file.
                                         Defaults to "point_cloud_animation.gif".
        camera_frustum_size (float, optional): The size of the camera frustum visualization.
                                               Defaults to 0.1.
    """
    points_transformed = points
    plotter = pv.Plotter(off_screen=True, window_size=[1280, 720])
    plotter.open_gif(output_filename, fps=4)

    cumulative_points = []
    cumulative_colors = []

    for i in range(0, num_frames):
        print(f"Processing frame {i + 1}/{num_frames}")

        # --- Data filtering and accumulation (no changes here) ---
        current_points = points_transformed[i].reshape(-1, 3)
        current_colors = colors[i].reshape(-1, 3)
        current_confidence = confidence[i].flatten()
        mask = current_confidence > conf_thresh
        mask2 = current_points[:,2]<=1.5
        filtered_points = current_points[mask&mask2]
        filtered_colors = current_colors[mask&mask2]

        if filtered_points.size > 0:
            cumulative_points.append(filtered_points)
            cumulative_colors.append(filtered_colors)

        plotter.clear()
        top_down_set = False 

        if cumulative_points:
            all_points = np.vstack(cumulative_points)
            all_colors = np.vstack(cumulative_colors)
            point_cloud = pv.PolyData(all_points)
            point_cloud["colors"] = all_colors
            plotter.add_mesh(
                point_cloud,
                point_size=1,
                scalars="colors",
                rgb=True,
                ambient=1.0,
                show_edges=False, 
                lighting=False,
            )

        # --- CORRECTED CAMERA VISUALIZATION ---
        # Instead of pv.Camera, we manually create a pyramid mesh
        
        for j in range(i+1):
            # 1. Define the vertices of a simple pyramid at the origin
            apex = [0, 0, 0]
            # Base points of the pyramid, scaled by frustum_size
            base_pts = [
                [-1.0 * camera_frustum_size, -1.0 * camera_frustum_size, 2.0 * camera_frustum_size],
                [ 1.0 * camera_frustum_size, -1.0 * camera_frustum_size, 2.0 * camera_frustum_size],
                [ 1.0 * camera_frustum_size,  1.0 * camera_frustum_size, 2.0 * camera_frustum_size],
                [-1.0 * camera_frustum_size,  1.0 * camera_frustum_size, 2.0 * camera_frustum_size],
            ]
            frustum_points = np.array([apex] + base_pts)

            # 2. Define the faces of the pyramid (4 triangles)
            frustum_faces = np.array([
                [3, 0, 1, 2],  # Apex, Pt1, Pt2
                [3, 0, 2, 3],  # Apex, Pt2, Pt3
                [3, 0, 3, 4],  # Apex, Pt3, Pt4
                [3, 0, 4, 1],  # Apex, Pt4, Pt1
            ])

            # 3. Create the mesh
            frustum_mesh = pv.PolyData(frustum_points, faces=frustum_faces)
            
            # 4. Apply the camera pose transformation to the pyramid mesh
            camera_pose = camera_poses[j]
            camera_pose = camera_pose@np.linalg.inv(np.array([
                [0,-1,0,0],
                [0,0,-1,0],
                [1,0,0,0],
                [0,0,0,1],
            ]))
            frustum_mesh.transform(camera_pose)

            plotter.add_mesh(frustum_mesh, color="blue", style="wireframe", line_width=3)
            # --- END OF CORRECTION ---

        if not top_down_set:
            xyz_min, xyz_max = points.reshape((-1,3)).min(0), points.reshape((-1,3)).max(0)
            centre = points.reshape((-1,3)).mean(axis=0)

            # pick a distance that comfortably encloses the cloud
            radius = 25

            elev_deg   = 80                                   # <-- change here
            azim_deg   = 0                                   # 45° just looks nice; tweak if you like
            elev_rad   = np.deg2rad(elev_deg)
            azim_rad   = np.deg2rad(azim_deg)

            # Spherical → Cartesian
            cam_x = radius * np.cos(elev_rad) * np.cos(azim_rad)
            cam_y = radius * np.cos(elev_rad) * np.sin(azim_rad)
            cam_z = radius * np.sin(elev_rad)

            plotter.camera_position = [
                (centre[0] + cam_x, centre[1] + cam_y, centre[2] + cam_z),  # camera
                tuple(centre),                                             # focal point
                (0, 0, 1)                                                  # up-vector (Z)
            ]
            top_down_set = True

        plotter.write_frame()

    plotter.close()
    print(f"\nAnimation saved to {output_filename}")

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    npz_file_path = './scene_output.npz'
    json_file_path = './sampled_big_room_undistort/transforms.json'

    data = np.load(npz_file_path)

    point_cloud_vertices = data.get('point_cloud')
    point_cloud_colors = data.get('point_cloud_colors')

    extrinsics = data.get('extrinsics')
    world_to_camera_vggt = np.zeros((extrinsics.shape[0],4,4))
    world_to_camera_vggt[:,:3,:] = extrinsics 
    world_to_camera_vggt[:,3,3] = 1
    camera_to_world_vggt = np.linalg.inv(world_to_camera_vggt)

    with open(json_file_path,'r') as f:
        json_data = json.load(f)
    frames = json_data['frames']
    all_transform = []
    for frame in frames:
        all_transform.append(frame['transform_matrix'])
    camera_to_world_json = np.array(all_transform)
    trans_off = np.zeros((4,4))    
    trans_off[:3,:3] = Rotation.from_euler('xyz',[0,-19,0], degrees=True).as_matrix()
    trans_off[3,3] = 1
    camera_to_world_json = camera_to_world_json@trans_off

    refl_matrix = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]
    ])
    camera_to_world_vggt = camera_to_world_vggt @ refl_matrix
    T_hat, s_hat = solve_T_s(camera_to_world_vggt[:10], camera_to_world_json[:10])
    print(T_hat, s_hat)

    camera_to_world_vggt_transformed = a_to_b(camera_to_world_vggt, T_hat, s_hat)          # reconstruct b from a

    points_all = data.get('raw_point_cloud')
    colors_all = data.get('raw_point_color')
    conf_all = data.get('raw_point_conf')

    num_frames, width, height, _ = points_all.shape
    points_flat = points_all.reshape(-1, 3)
    transformed_points_flat = transform_point_cloud(points_flat, T_hat, s_hat)
    points_transformed = transformed_points_flat.reshape(num_frames, width, height, 3)
    
    # orig_shape = points_all.shape
    # points_all = np.reshape(points_all, (orig_shape[0], orig_shape[1]*orig_shape[2], orig_shape[3]))
    # colors_all = np.reshape(colors_all, (orig_shape[0], orig_shape[1]*orig_shape[2], orig_shape[3]))
    # conf_all = np.reshape(conf_all, (orig_shape[0], orig_shape[1]*orig_shape[2], orig_shape[3]))
    conf_tmp = np.reshape(conf_all, (-1))
    conf_thresh = np.percentile(conf_tmp, 60)

    create_point_cloud_animation(
        points_transformed,
        colors_all,
        camera_to_world_vggt_transformed,
        conf_all,
        conf_thresh,
        output_filename="cool_animation.gif"
    )