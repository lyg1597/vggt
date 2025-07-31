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

from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import profiler
# from memory_profiler import profile

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA is not available. Running on CPU will be very slow.")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded successfully.")
    return model, device

# --- Step 2: Image Preprocessing ---
def preprocess_input_images(image_dir, image_filenames, device: str) -> Optional[torch.Tensor]:
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
        images = load_and_preprocess_images(image_paths).to(device)
        return images
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Step 3: Model Inference ---
def run_model_inference(model: VGGT, images: torch.Tensor) -> Dict[str, Any]:
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.cuda.memory._record_memory_history(
        max_entries=100000
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
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
    images_shape: tuple,
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
            predictions_np[key] = raw_predictions[key].cpu().numpy().squeeze(0)

    depth_map = predictions_np["depth"]
    extrinsics = predictions_np['extrinsic']
    intrinsics = predictions_np['intrinsic']

    print("Computing world points from depth map...")
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsics, intrinsics)
    predictions_np["world_points_from_depth"] = world_points

    # --- Start of new color logic ---
    # The 'images' tensor is still in the raw_predictions dict from the model output
    images_np = raw_predictions["images"].cpu().numpy().squeeze(0)

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
def visualize_point_cloud(vertices: np.ndarray, colors: np.ndarray):
    """
    Creates an interactive 3D plot of the colored point cloud using PyVista.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 3) for point coordinates.
        colors (np.ndarray): A NumPy array of shape (N, 3) for RGB colors.
    """
    if vertices.size == 0:
        print("Cannot visualize: The point cloud is empty.")
        return
    print("Opening visualization window...")

    cloud = pv.PolyData(vertices)
    # Add the color data to the PolyData object as "scalars"
    cloud["colors"] = colors

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=5,
        scalars="colors",  # Tell PyVista to use the color data
        rgb=True,          # Specify that the scalars are RGB colors
    )
    plotter.add_axes()
    plotter.enable_eye_dome_lighting()
    plotter.set_background('black')
    print("Close the PyVista window to exit the script.")
    plotter.show()

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
    
def is_close(transform1, transform2):
    # if np.linalg.norm(transform1[:,3]-transform2[:,3])>0.1 or \
    #     np.linalg.norm(transform1[:3,:3]-transform2[:3,:3]) > 0.25:
    #     return False 
    # else:
    #     return True 
    return False

def main(args):
    model, device = load_vggt_model()
    transform_json_fn = os.path.join(args.image_dir, 'transforms.json')
    with open(transform_json_fn, 'r') as f:
        transform_json = json.load(f)
    frames = transform_json['frames']
    # extrinsics = []
    filenames = []
    point_cloud = None
    prev_transform = None
    idx = 0 
    for frame_idx in range(len(frames)):
    # for frame_idx in range(200):
        # if idx>=2:
        #     break
        frame = frames[frame_idx]
        transform_matrix = np.array(frame['transform_matrix'])
        image_fn = frame['file_path']
        if prev_transform is not None and is_close(transform_matrix, prev_transform):
            continue
        else:
            filenames.append(image_fn)
        #     # prev_transform = transform_matrix
        if len(filenames)<args.num_batch:
            continue 
        print(frame_idx)
        images = preprocess_input_images(args.image_dir, filenames, device)
        raw_predictions = run_model_inference(model, images)
        final_results = extract_and_filter_scene(raw_predictions, images.shape, args.conf_thres, args.branch)
        point_cloud, pcd_compressed = concate_results(point_cloud, final_results)

        # np.savez_compressed(
        #     f'step_res/res_{idx:05d}.npz',
        #     point_cloud = point_cloud
        # )
        o3d.io.write_point_cloud(
            f'step_res/res_{idx:05d}.ply', 
            pcd_compressed
        )

        idx += 1
        filenames = filenames[-40:]

    # extrinsics_matrices = []
    # frames_fns = []
    # for i in range(len(extrinsics)):
    #     extrinsics_matrices.append(extrinsics[i][1])
    #     frames_fns.append(extrinsics[i][0])
    np.savez_compressed(
        args.output_file,
        point_cloud = point_cloud
    )
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
