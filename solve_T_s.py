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

# --- Simplified visualization function ---
def visualize_point_cloud(plotter, points, colors):
    """
    Creates an interactive 3D plot of the transformed point cloud.

    Args:
        points (np.ndarray): The point cloud in the JSON frame.
        colors (np.ndarray): The RGB colors for the points.
    """
    print("Opening visualization window...")
    cloud = pv.PolyData(points)
    cloud["colors"] = colors

    # plotter = pv.Plotter(window_size=[1200, 800])
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=5,
        scalars="colors",
        rgb=True,
        ambient=1.0
    )
    plotter.add_title("Transformed Point Cloud (JSON Frame)", font_size=12)
    plotter.enable_eye_dome_lighting()
    plotter.add_axes()
    print("Close the PyVista window to exit the script.")
    # plotter.show()
    return plotter

def visualize_cameras(
        fig, 
        camera_poses,
        x_color = 'red',
        y_color = 'green',
        z_color = 'blue',
        line_length = 0.5,
        line_width = 4,
        marker_color = 'red',
        marker_size = 0.01,
    ):
    for i in range(camera_poses.shape[0]):           # rgb_path unused here

        # Full camera rotation with –5° pitch offset applied
        pos = camera_poses[i,:3,3]
        R_final = camera_poses[i,:3,:3]

        # Local axes endpoints
        x_end = pos + R_final @ np.array([line_length, 0, 0])
        y_end = pos + R_final @ np.array([0, line_length, 0])
        z_end = pos + R_final @ np.array([0, 0, line_length])

        # Draw camera-frame axes (short lines)
        fig.add_mesh(pv.Line(pos, x_end), color=x_color,   line_width=line_width)
        fig.add_mesh(pv.Line(pos, y_end), color=y_color, line_width=line_width)
        fig.add_mesh(pv.Line(pos, z_end), color=z_color,  line_width=line_width)

        # Red point marking the camera position
        fig.add_mesh(pv.Sphere(radius=marker_size, center=pos), color=marker_color)

    return fig

def save_point_cloud_to_ply(filename, points, colors):
    """
    Saves a colored point cloud to a PLY file.

    Args:
        filename (str): The path to save the PLY file.
        points (np.ndarray): The point cloud vertices (N, 3).
        colors (np.ndarray): The point cloud colors (N, 3).
    """
    print(f"Saving point cloud to {filename}...")
    # Create a trimesh PointCloud object
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    # Export to PLY format
    cloud.export(file_obj=filename, file_type='ply')
    print("Successfully saved.")

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

    refl_matrix = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]
    ])
    camera_to_world_vggt = camera_to_world_vggt @ refl_matrix
    T_hat, s_hat = solve_T_s(camera_to_world_vggt, camera_to_world_json)
    print(T_hat, s_hat)

    camera_to_world_vggt_transformed = a_to_b(camera_to_world_vggt, T_hat, s_hat)          # reconstruct b from a
    # a_pred = b_to_a(b, T_hat, s_hat)          # reconstruct a from b
    # refl_matrix = np.eye(4)

    # quick sanity-check
    rot_err  = np.max(np.linalg.norm(camera_to_world_vggt_transformed[:, :3, :3] - camera_to_world_json[:, :3, :3], axis=(1, 2)))
    trans_err = np.max(np.linalg.norm(camera_to_world_vggt_transformed[:, :3, 3] - camera_to_world_json[:, :3, 3], axis=1))
    print(f"max orientation error: {rot_err:.3e}")
    print(f"max translation error: {trans_err:.3e}")
    print(np.linalg.norm(camera_to_world_vggt_transformed[:, :3, :3] - camera_to_world_json[:, :3, :3], axis=(1, 2)))

    transformed_point_cloud = transform_point_cloud(point_cloud_vertices, T_hat, s_hat)

    fig = pv.Plotter()
    fig.set_background('white')
    mask = transformed_point_cloud[:,2]<1.5
    transformed_point_cloud_masked = transformed_point_cloud[mask]
    point_cloud_colors_masked = point_cloud_colors[mask]

    fig = visualize_point_cloud(fig, transformed_point_cloud_masked, point_cloud_colors_masked)
    # Global XYZ reference axes (world frame)
    fig.add_mesh(pv.Line((0, 0, 0), (10, 0, 0)), color="red",   line_width=4)
    fig.add_mesh(pv.Line((0, 0, 0), (0, 10, 0)), color="green", line_width=4)
    fig.add_mesh(pv.Line((0, 0, 0), (0, 0, 10)), color="blue",  line_width=4)

    fig = visualize_cameras(fig, camera_to_world_json, x_color = 'blue', y_color='blue', z_color='blue', marker_color='blue', line_length=0.25, marker_size=0.05)

    fig = visualize_cameras(fig, camera_to_world_vggt_transformed, x_color = 'red', y_color='red', z_color='red', marker_color='red', line_length=0.25, marker_size=0.05)

    fig.show()

    save_point_cloud_to_ply('output_point_cloud_transformed.ply', transformed_point_cloud, point_cloud_colors)