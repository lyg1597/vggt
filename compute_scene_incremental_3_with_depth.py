import os
import glob
import torch
import numpy as np
import argparse
import sys
from typing import Dict, Any, Optional
import json 

import pyvista as pv
import pandas as pd
from scipy.spatial.transform import Rotation

from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import profiler
from solve_T_s import solve_T_s, a_to_b, transform_point_cloud_a_to_b
# from memory_profiler import profile
from PIL import Image 

sys.path.append(os.path.abspath(".")) 

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# from solve_T_s import solve_T_s, a_to_b, transform_point_cloud_a_to_b
import open3d as o3d


# --- Step 1: Model Loading ---
def load_vggt_model() -> (VGGT, str):
    print("Initializing and loading VGGT model...")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA is not available. Running on CPU will be very slow.")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(dtype).to(device)
    print("Model loaded successfully.")
    return model, device, dtype

# --- Step 2: Image Preprocessing ---
def preprocess_input_images(image_dir, image_filenames, device: str, dtype) -> Optional[torch.Tensor]:
    # image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    # image_paths = []
    # for ext in image_extensions:
    #     image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    # if not image_paths:
    #     print(f"Error: No images found in directory: {image_dir}")
    #     return None
    # image_paths = sorted(image_paths)
    image_paths = []
    for i in range(len(image_filenames)):
        image_paths.append(os.path.join(image_dir, image_filenames[i]))
    print(f"Found {len(image_paths)} images. Preprocessing...")
    try:
        images = load_and_preprocess_images(image_paths).to(dtype).to(device)
        return images
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Step 3: Model Inference ---
def run_model_inference(model: VGGT, images: torch.Tensor) -> Dict[str, Any]:
    print("Running model inference...")
    torch.cuda.memory._record_memory_history(
        max_entries=100000
    )
    with torch.no_grad():
        # with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)
    print("Inference complete.")

    torch.cuda.memory._dump_snapshot(f"memory_res.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    # Compute extrinsics and intrinsics from pose encoding (still as tensors for now)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    return predictions

# --- Step 4: Post-processing and Filtering ---
def extract_and_filter_scene(
    raw_predictions: Dict[str, Any],
    conf_thres_percent: float,
    branch: str
) -> Dict[str, np.ndarray]:
    """
    Processes raw model predictions to compute extrinsics, intrinsics, and a
    filtered 3D point cloud with colors.
    """
    print("Extracting camera parameters and filtering point cloud...")

    # Convert tensors to numpy
    predictions_np = {}
    for key in raw_predictions.keys():
        if isinstance(raw_predictions[key], torch.Tensor):
            predictions_np[key] = raw_predictions[key].cpu().numpy().squeeze(0).astype('float')

    depth_map = predictions_np["depth"]
    extrinsics = predictions_np['extrinsic']
    intrinsics = predictions_np['intrinsic']

    print("Computing world points from depth map...")
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)
    predictions_np["world_points_from_depth"] = world_points

    # --- Start of new color logic ---
    # The 'images' tensor is still in the raw_predictions dict from the model output
    images_np = raw_predictions["images"].cpu().numpy().squeeze(0).astype('float')

    # Handle different image formats (NCHW vs NHWC)
    # This logic is from visual_util.py
    if images_np.ndim == 4 and images_np.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images_np, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images_np

    # Reshape colors to a flat list and scale to 0-255
    colors_rgb_reshaped = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)
    # --- End of new color logic ---

    if branch == 'pointmap':
        print("Using Pointmap Branch for point cloud.")
        if "world_points" in predictions_np:
            pred_world_points = predictions_np["world_points"]
            pred_world_points_conf = predictions_np.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: 'world_points' not in predictions. Falling back to depth branch.")
            pred_world_points = predictions_np["world_points_from_depth"]
            pred_world_points_conf = predictions_np.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else: # branch == 'depth'
        print("Using Depth Branch for point cloud.")
        pred_world_points = predictions_np["world_points_from_depth"]
        pred_world_points_conf = predictions_np.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)

    conf_threshold = np.percentile(conf, conf_thres_percent) if conf_thres_percent > 0 else 0.0
    conf_mask = (conf >= conf_threshold)

    # Apply the same mask to both vertices and colors
    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_reshaped[conf_mask]

    print(f"Filtered point cloud and colors from {len(vertices_3d)} to {len(filtered_vertices)} points.")

    # Return colors in the final dictionary
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_cloud": filtered_vertices,
        "point_cloud_colors": filtered_colors,
        "raw_point_cloud": pred_world_points,
        "raw_point_conf": pred_world_points_conf,
        "raw_point_color": colors_rgb,
    }

# --- Step 5: Visualization ---
def visualize_point_cloud(plotter, points, colors):
    """
    Creates an interactive 3D plot of the colored point cloud using PyVista.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 3) for point coordinates.
        colors (np.ndarray): A NumPy array of shape (N, 3) for RGB colors.
    """
    print("Opening visualization window...")
    cloud = pv.PolyData(points)
    cloud["colors"] = colors

    # plotter = pv.Plotter(window_size=[1200, 800])
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=1,
        scalars="colors",
        rgb=True,
        ambient=1.0,
        show_edges=False, 
        lighting=False,
    )
    plotter.add_title("Transformed Point Cloud (JSON Frame)", font_size=12)
    # plotter.enable_eye_dome_lighting()
    plotter.add_axes()
    print("Close the PyVista window to exit the script.")
    # plotter.show()
    return plotter

def align_data(new_data, base_data=None):
    """
    Aligns a new point cloud to a base point cloud using Fast Global Registration.

    If the base_data is None, this function acts as an initialization and
    simply returns the new_data. Otherwise, it aligns the new_data (source)
    to the base_data (target).

    Args:
        new_data (np.ndarray): The source point cloud data as an Nx6 NumPy array.
        base_data (np.ndarray, optional): The target point cloud. Defaults to None.

    Returns:
        np.ndarray: A combined NumPy array of the aligned point clouds.
    """
    if base_data is None:
        print("--- Base data is None. Returning new data as is. ---")
        return new_data

    print("--- Starting Point Cloud Alignment ---")

    # 1. Convert NumPy arrays to Open3D PointCloud objects
    print("1. Converting NumPy arrays to Open3D PointCloud objects...")
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(base_data[:, :3])
    target.colors = o3d.utility.Vector3dVector(base_data[:, 3:])
    
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(new_data[:, :3])
    source.colors = o3d.utility.Vector3dVector(new_data[:, 3:])

    # --- KEY PARAMETER TO TUNE ---
    voxel_size = 0.05 
    
    # 2. Prepare Point Clouds for FGR
    print("\n2. Downsampling and preparing features for FGR...")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 3. Perform Fast Global Registration
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
    
    # 4. Combine clouds and convert back to NumPy
    print("\n4. Combining point clouds...")
    combined_pcd = target + source
    
    combined_points = np.asarray(combined_pcd.points)
    combined_colors = np.asarray(combined_pcd.colors)
    combined_data = np.hstack((combined_points, combined_colors))

    print("\n--- Alignment Complete ---")
    return combined_data

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

def concate_results(base_point_cloud, new_result):
    '''
    The function takes point cloud result1 and result2 and concatenate them together
    The concatenation can only happen when there's overlapping frames between result1 and result2
    The index of overlapping frames in result1 is specified by overlap
    The overlapping frames in result2 is specified by result2[:len(overlap)]

    :param result1: The first point cloud
    :type result1: Dict
    :param result2: The second point cloud
    :type result2: Dict
    :param overlap: The index of overlapping frames
    :type result2: List[int]
    ...
    :return: The concatenated point clouds
    :rtype: Dict
    '''
    point_data = new_result['point_cloud'].reshape((-1,3))
    point_colors = new_result['point_cloud_colors'].reshape((-1,3))

    # FIX 1: Normalize colors from 0-255 to the required 0.0-1.0 range.
    # We also ensure the data type is float before stacking.
    point_colors_normalized = point_colors.astype(np.float64) / 255.0

    new_pcd = np.hstack((point_data, point_colors_normalized))
    aligned_pcd = align_data(new_pcd, base_point_cloud)
    
    # FIX 2: Correctly handle the two return values from compress_data.
    compressed_pcd_array, compressed_pcd_obj = compress_data(aligned_pcd)
    
    # Return both the array (for the next iteration's base_point_cloud)
    # and the o3d object (for saving to .ply).
    return compressed_pcd_array, compressed_pcd_obj
    
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
        # refl_matrix = np.array([
        #     [0,-1,0],
        #     [0,0,-1],
        #     [1,0,0],
        # ])
        pos = camera_poses[i,:3,3]
        # R_final = camera_poses[i,:3,:3]@np.linalg.inv(refl_matrix)
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

def map_result(gt_extrinsics, raw_predictions, args = None):
    final_results = extract_and_filter_scene(raw_predictions, args.conf_thres, args.branch)

    predictions_np = {}
    for key in raw_predictions.keys():
        if isinstance(raw_predictions[key], torch.Tensor):
            predictions_np[key] = raw_predictions[key].cpu().numpy().squeeze(0).astype('float')

    depth_map = predictions_np["depth"]
    extrinsics = predictions_np['extrinsic']
    intrinsics = predictions_np['intrinsic'] 
    intrinsic = np.mean(intrinsics, axis=0)

    world_to_camera_vggt = np.zeros((extrinsics.shape[0],4,4))
    world_to_camera_vggt[:,:3,:] = extrinsics 
    world_to_camera_vggt[:,3,3] = 1
    camera_to_world_vggt = np.linalg.inv(world_to_camera_vggt)

    camera_to_world_json = np.array(gt_extrinsics)
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

    # camera_to_world_vggt = camera_to_world_vggt @ refl_matrix
    T_hat, s_hat = solve_T_s(camera_to_world_vggt[:], camera_to_world_json[:])
    print(T_hat, s_hat)

    camera_to_world_vggt_transformed = a_to_b(camera_to_world_vggt, T_hat, s_hat)
    rotation_vggt_transformed = camera_to_world_vggt_transformed[:,:3,:3]
    # position_vggt_transformed = camera_to_world_vggt_transformed[:,:3,3]
    refl_matrix = np.array([
        [0,-1,0],
        [0,0,-1],
        [1,0,0],
    ])
    rotation_vggt_transformed = rotation_vggt_transformed@np.linalg.inv(refl_matrix)
    camera_to_world_vggt_transformed[:,:3,:3] = rotation_vggt_transformed
    depth_map_transformed = depth_map*abs(s_hat) 

    if args.visualize:
        
        fig = pv.Plotter()
        fig.set_background('white')
        rotation_json = camera_to_world_json[:,:3,:3]
        refl_matrix = np.array([
            [0,-1,0],
            [0,0,-1],
            [1,0,0],
        ])
        rotation_json = rotation_json@np.linalg.inv(refl_matrix)     
        camera_to_world_json[:,:3,:3] = rotation_json   
        fig = visualize_cameras(fig, camera_to_world_json, x_color = 'red', y_color='green', z_color='blue', marker_color='blue', line_length=0.25, marker_size=0.05)
        fig = visualize_cameras(fig, camera_to_world_vggt_transformed, x_color = 'purple', y_color='purple', z_color='purple', marker_color='red', line_length=0.25, marker_size=0.05)
        
        final_results = extract_and_filter_scene(raw_predictions, args.conf_thres, args.branch)
        point_cloud_vertices = final_results['point_cloud']
        point_cloud_colors = final_results['point_cloud_colors']

        transformed_point_cloud = transform_point_cloud_a_to_b(point_cloud_vertices, T_hat, s_hat)

        fig = visualize_point_cloud(fig, transformed_point_cloud, point_cloud_colors)

        # fig = visualize_cameras(fig, camera_to_world_vggt, x_color = 'blue', y_color='blue', z_color='blue', marker_color='blue', line_length=0.25, marker_size=0.05)
        fig.show()

    return camera_to_world_vggt_transformed, depth_map_transformed, intrinsic, T_hat, s_hat 

def main(args):
    model, device, dtype = load_vggt_model()
    transform_json_fn = os.path.join(args.image_dir, 'transforms.json')
    with open(transform_json_fn, 'r') as f:
        transform_json = json.load(f)
    frames = transform_json['frames']
    # extrinsics = []
    filenames = []
    gt_extrinsics = []
    point_cloud = None
    prev_transform = None
    idx = 0 
    prev_idx = 0
    res_depth_idx = 0
    for frame_idx in range(35, len(frames)):
    # for frame_idx in range(200):
        # if idx>=10:
        #     break
        frame = frames[frame_idx]
        transform_matrix = np.array(frame['transform_matrix'])
        gt_extrinsics.append(transform_matrix)
        image_fn = frame['file_path']
        # if prev_transform is not None and is_close(transform_matrix, prev_transform):
        #     continue
        # else:
        filenames.append(image_fn)
        #     # prev_transform = transform_matrix
        if len(filenames)<args.num_batch:
            continue 
        print(frame_idx)
        images = preprocess_input_images(args.image_dir, filenames, device, dtype)
        raw_predictions = run_model_inference(model, images)
        mapped_c2w, mapped_depth, intrinsic, T, s = map_result(gt_extrinsics, raw_predictions, args)

        for j in range(prev_idx, mapped_c2w.shape[0]):
            img_fn = f'step_res_depth/frame_{res_depth_idx:05d}.png'
            depth_fn = f'step_res_depth/frame_depth_{res_depth_idx:05d}.png'
            rgb = torch.permute(images[j], dims=(1,2,0)).detach().cpu().numpy()
            depth = mapped_depth[j]
            if not args.no_dump:
                np.savez(
                    f'step_res_depth/res_img_depth_{res_depth_idx:05d}.npz',
                    c2w = mapped_c2w[j],
                    intrinsic = intrinsic,
                    rgb = rgb,
                    depth = np.clip(depth/10.0, 0, 1),
                    img_fn = img_fn,
                    depth_fn = depth_fn,
                    depth_scale = 10.0
                )
                image_pil = Image.fromarray((rgb*255).astype(np.uint8))
                image_pil.save(img_fn)
                depth_pil = Image.fromarray((np.clip(depth.squeeze()/10.0, 0, 1)*255).astype(np.uint8))
                depth_pil.save(depth_fn)
                res_depth_idx += 1
        # final_results = extract_and_filter_scene(raw_predictions, images.shape, args.conf_thres, args.branch)
        # np.savez(
        #     f'step_res/res_raw_{idx:05d}.npz',
        #     extrinsics=final_results['extrinsics'],
        #     intrinsics=final_results['intrinsics'],
        #     point_cloud=final_results['point_cloud'],
        #     point_cloud_colors = final_results['point_cloud_colors'],
        #     # raw_point_cloud = final_results["raw_point_cloud"],
        #     # raw_point_conf = final_results["raw_point_conf"],
        #     # raw_point_color = final_results["raw_point_color"],
        #     images = torch.permute(images, dims=((0,2,3,1))).detach().cpu().numpy(),
        # )
        # point_cloud, pcd_compressed = concate_results(point_cloud, final_results)

        # np.savez_compressed(
        #     f'step_res/res_{idx:05d}.npz',
        #     point_cloud = point_cloud
        # )
        # o3d.io.write_point_cloud(
        #     f'step_res/res_{idx:05d}.ply', 
        #     pcd_compressed
        # )

        # points_array = np.asarray(pcd_compressed.points)
        # np.save(f'step_res/res_geo_{idx:05d}.npy', points_array)


        idx += 1
        if len(filenames)>70:
            filenames = []
            gt_extrinsics = []
        else:
            filenames = filenames[-int(len(filenames)*2.0/3.0):]
            gt_extrinsics = gt_extrinsics[-int(len(gt_extrinsics)*2.0/3.0):]
        prev_idx = len(filenames)

    # extrinsics_matrices = []
    # frames_fns = []
    # for i in range(len(extrinsics)):
    #     extrinsics_matrices.append(extrinsics[i][1])
    #     frames_fns.append(extrinsics[i][0])
    # np.savez_compressed(
    #     args.output_file,
    #     point_cloud = point_cloud
    # )
    # o3d.io.write_point_cloud(
    #     f'step_res/res_{idx:05d}.npz', 
    #     pcd_compressed
    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VGGT to compute camera parameters and a point cloud from images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--output_file", type=str, default="scene_output.npz", help="Path to save the output .npz file.")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="Confidence threshold percentile (0-100) for filtering points.")
    parser.add_argument(
        "--branch", 
        type=str, 
        default="pointmap", 
        choices=['pointmap', 'depth'],
        help="Select the source for the point cloud: 'pointmap' (direct regression) or 'depth' (unprojected depth map)."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, display the point cloud in an interactive window after computation."
    )
    parser.add_argument(
        "--num_batch",
        type = int,
        default = 35,
        help="number of frames in the batch"
    )
    parser.add_argument(
        "--no_dump",
        action="store_true",
        help="If set, disable saving result to files."
    )    

    
    args = parser.parse_args()

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             profile_memory=True,
    #             record_shapes=True,
    #             with_stack=True) as prof:
    #     with record_function("model_inference"): # Optional label for the code block
    # main(args)

    # Wrap the code you want to profile with emit_nvtx()
    main(args)

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_memory_usage", row_limit=5))
