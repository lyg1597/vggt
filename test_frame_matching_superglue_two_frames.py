import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.cm as cm

# --- SuperGlue Imports ---
# Add the SuperGluePretrainedNetwork directory to your python path
# Or run this script from the same directory.
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (
    frame2tensor, make_matching_plot_fast)

def match_with_superglue(img0, img1, device='cpu', visualize=False):
    """
    Matches two images using the SuperPoint and SuperGlue models.

    Args:
        img0 (np.array): The first image (BGR format).
        img1 (np.array): The second image (BGR format).
        device (str): The device to run the models on ('cpu' or 'cuda').
        visualize (bool): If True, displays the matches.

    Returns:
        tuple: A tuple containing (mkpts0, mkpts1, mconf).
            - mkpts0 (np.array): Matched keypoints from image 0 (N, 2).
            - mkpts1 (np.array): Matched keypoints from image 1 (N, 2).
            - mconf (np.array): Confidence scores for each match (N,).
    """
    # --- Model Configuration ---
    # These settings are recommended by the authors.
    superglue_weights = 'outdoor' # 'indoor' or 'outdoor'
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1 # -1 for no limit
        },
        'superglue': {
            'weights': superglue_weights,
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    # 1. Initialize the Matching model
    print("Initializing SuperGlue model...")
    matching = Matching(config).eval().to(device)
    print(f"Model loaded on device: {device}")

    # 2. Preprocess the images
    # SuperGlue expects grayscale images as tensors.
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    tensor0 = frame2tensor(img0_gray, device)
    tensor1 = frame2tensor(img1_gray, device)

    # 3. Perform the matching
    with torch.no_grad():
        pred = matching({'image0': tensor0, 'image1': tensor1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # 4. Extract keypoints and matches
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    # Filter out invalid matches (-1)
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    # 5. Visualize if requested
    if visualize:
        color = cm.jet(mconf)
        viz_img = make_matching_plot_fast(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kpts0, kpts1, mkpts0, mkpts1, color, '')
            
        # Resize for display
        h, w, _ = viz_img.shape
        if w > 1920:
            scale = 1920 / w
            h_new, w_new = int(h * scale), int(w * scale)
            viz_img = cv2.resize(viz_img, (w_new, h_new))

        cv2.imshow('SuperGlue Matches', viz_img)
        print("Displaying SuperGlue visualization. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyWindow('SuperGlue Matches')

    return mkpts0, mkpts1, mconf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue feature matching demo for a pair of images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image_path1', type=str, help='Path to the first image.')
    parser.add_argument('image_path2', type=str, help='Path to the second image.')
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cpu' or 'cuda')")
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of matches.')
    args = parser.parse_args()

    # Check if SuperGlue repo exists
    if not Path('SuperGluePretrainedNetwork').exists():
        print("Error: 'SuperGluePretrainedNetwork' directory not found.")
        print("Please clone the repository: git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git")
        exit()
        
    # Load the two images
    try:
        img0 = cv2.imread(args.image_path1)
        img1 = cv2.imread(args.image_path2)
        if img0 is None or img1 is None:
            raise IOError
    except (IOError, FileNotFoundError):
        print(f"Error: Could not read images. Check paths.")
        exit()

    print("--- Running SuperGlue Matching ---")
    
    # Run the matching function with visualization enabled
    matched_kpts0, matched_kpts1, match_confidences = match_with_superglue(
        img0, img1, device=args.device, visualize=args.visualize)

    print(f"\nFound {len(matched_kpts0)} matches.")
    print("--- Demo Finished ---")
