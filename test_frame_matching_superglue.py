import torch
import cv2
import numpy as np
import argparse
import os
import glob
from pathlib import Path
import matplotlib.cm as cm
# --- SuperGlue Imports ---
# Add the SuperGluePretrainedNetwork directory to your python path
# Or run this script from the same directory.
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (
    frame2tensor, make_matching_plot_fast)

def sequential_match_superglue(images, device='cpu', visualize=False):
    """
    Performs sequential feature matching on a list of images using SuperGlue.

    Args:
        images (list): A list of images as NumPy arrays (in BGR format).
        device (str): The device to run the models on ('cpu' or 'cuda').
        visualize (bool): If True, displays the matches for each pair.

    Returns:
        list: A list of dictionaries. Each dictionary contains the matches
              for a pair, with keys 'mkpts0', 'mkpts1', and 'mconf'.
    """
    if len(images) < 2:
        print("Error: At least two images are required for matching.")
        return []

    # --- Model Configuration ---
    superglue_weights = 'outdoor'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    # 1. Initialize the Matching model once
    print("Initializing SuperGlue model...")
    matching = Matching(config).eval().to(device)
    print(f"Model loaded on device: {device}")

    all_matches = []
    
    # 2. Preprocess all images
    print("Preprocessing all images...")
    tensors = [frame2tensor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), device) for img in images]

    # 3. Loop through consecutive pairs
    print("\nMatching sequential pairs...")
    for i in range(len(images) - 1):
        print(f"  - Matching image {i} with image {i+1}...")
        img0, img1 = images[i], images[i+1]
        tensor0, tensor1 = tensors[i], tensors[i+1]
        
        # Perform the matching
        with torch.no_grad():
            pred = matching({'image0': tensor0, 'image1': tensor1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        # Extract keypoints and matches
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Filter out invalid matches (-1)
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        
        all_matches.append({'mkpts0': mkpts0, 'mkpts1': mkpts1, 'mconf': mconf})
        print(f"    -> Found {len(mkpts0)} matches.")

        # Visualize if requested
        if visualize:
            color = cm.jet(mconf)
            viz_img = make_matching_plot_fast(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kpts0, kpts1, mkpts0, mkpts1, color, '')
            
            h, w, _ = viz_img.shape
            if w > 1920:
                scale = 1920 / w
                h_new, w_new = int(h * scale), int(w * scale)
                viz_img = cv2.resize(viz_img, (w_new, h_new))

            cv2.imshow(f'SuperGlue Matches: {i} -> {i+1}', viz_img)
            print("      -> Displaying visualization. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow(f'SuperGlue Matches: {i} -> {i+1}')
            
    if visualize:
        print("\nVisualization complete.")

    return all_matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sequential SuperGlue feature matching on a folder of images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help='Path to the folder containing images.')
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cpu' or 'cuda')")
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of matches.')
    args = parser.parse_args()

    # Check if SuperGlue repo exists
    if not Path('SuperGluePretrainedNetwork').exists():
        print("Error: 'SuperGluePretrainedNetwork' directory not found.")
        print("Please clone the repository: git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git")
        exit()

    # Find and sort images
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.folder, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"Error: No images found in '{args.folder}'")
        exit()

    # Load images
    image_sequence = [cv2.imread(str(p)) for p in image_paths]
    print(f"--- Found and loaded {len(image_sequence)} images from '{args.folder}' ---")

    # Run the sequential matching function
    matches_list = sequential_match_superglue(
        image_sequence, device=args.device, visualize=args.visualize)

    print("\n--- Feature Matching Summary ---")
    for i, match_dict in enumerate(matches_list):
        num_matches = len(match_dict['mkpts0'])
        img_name0 = os.path.basename(image_paths[i])
        img_name1 = os.path.basename(image_paths[i+1])
        print(f"Total matches found between {img_name0} and {img_name1}: {num_matches}")
    
    print("\n--- Demo Finished ---")
