import os
import glob
import torch
import numpy as np
import argparse
import sys
from typing import Dict, Any, Optional

import pyvista as pv

from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import profiler
# from memory_profiler import profile

sys.path.append(os.path.abspath(".")) 

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

import json 
from scipy.spatial.transform import Rotation 
from solve_T_s import solve_T_s, a_to_b, transform_point_cloud_a_to_b
import time 

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
    model = model.to(torch.float16).to(device)
    print("Model loaded successfully.")
    return model, device

# --- Step 2: Image Preprocessing ---
def preprocess_input_images(image_dir: str, device: str) -> Optional[torch.Tensor]:
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    if not image_paths:
        print(f"Error: No images found in directory: {image_dir}")
        return None
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images. Preprocessing...")
    try:
        images = load_and_preprocess_images(image_paths).to(torch.float16).to(device)
        return images
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Step 3: Model Inference ---
def run_model_inference(model: VGGT, images: torch.Tensor) -> Dict[str, Any]:
    print("Running model inference...")
    # dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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

    conf_threshold = np.percentile(conf.astype(float), conf_thres_percent) if conf_thres_percent > 0 else 0.0
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

def reproject_final_results(image_dir, final_results):
    point_cloud_vertices = final_results['point_cloud']
    point_cloud_colors = final_results['point_cloud_colors']

    extrinsics = final_results['extrinsics']
    world_to_camera_vggt = np.zeros((extrinsics.shape[0],4,4))
    world_to_camera_vggt[:,:3,:] = extrinsics 
    world_to_camera_vggt[:,3,3] = 1
    camera_to_world_vggt = np.linalg.inv(world_to_camera_vggt)

    json_fn = os.path.join(image_dir, 'transforms.json')
    with open(json_fn, 'r') as f:
        transform_data = json.load(f)
    
    frames = transform_data['frames']

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
    T_hat, s_hat = solve_T_s(camera_to_world_vggt[:], camera_to_world_json[:])
    print(T_hat, s_hat)

    camera_to_world_vggt_transformed = a_to_b(camera_to_world_vggt, T_hat, s_hat)          # reconstruct b from a
    transformed_point_cloud = transform_point_cloud_a_to_b(point_cloud_vertices, T_hat, s_hat)
    # return camera_to_world_vggt_transformed, transformed_point_cloud
    final_results['point_cloud'] = transformed_point_cloud
    final_results['extrinsics'] = np.linalg.inv(camera_to_world_vggt_transformed)[:,:3,:]
    return final_results

def main(args):
    model, device = load_vggt_model()
    images = preprocess_input_images(args.image_dir, device)
    if images is None:
        return
    start_time = time.time()
    raw_predictions = run_model_inference(model, images)
    print("inference time:", time.time()-start_time)
    start_time = time.time()
    final_results = extract_and_filter_scene(raw_predictions, images.shape, args.conf_thres, args.branch)
    print("reconstruction time:", time.time()-start_time)
    
    if final_results:
        start_time = time.time()
        final_results = reproject_final_results(args.image_dir, final_results)
        print("reproject time:", time.time()-start_time)
    
        print("\n--- Computation Successful ---")
        print(f"Extrinsics shape: {final_results['extrinsics'].shape}")
        print(f"Intrinsics shape: {final_results['intrinsics'].shape}")
        print(f"Point Cloud shape: {final_results['point_cloud'].shape}")
        
        np.savez_compressed(
            args.output_file,
            extrinsics=final_results['extrinsics'],
            intrinsics=final_results['intrinsics'],
            point_cloud=final_results['point_cloud'],
            point_cloud_colors = final_results['point_cloud_colors'],
            raw_point_cloud = final_results["raw_point_cloud"],
            raw_point_conf = final_results["raw_point_conf"],
            raw_point_color = final_results["raw_point_color"],

        )
        print(f"\nResults saved to: {args.output_file}")
        
        if args.visualize:
            visualize_point_cloud(final_results['point_cloud'], final_results['point_cloud_colors'])


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
