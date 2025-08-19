import cv2
import numpy as np
import os
import argparse
import glob

def sequential_feature_matching(images, visualize=False):
    """
    Performs sequential feature matching on a list of images using ORB.

    This function takes a list of images and finds keypoint matches between
    each consecutive pair (image_i, image_i+1).

    Args:
        images (list): A list of images as NumPy arrays (in BGR format).
        visualize (bool): If True, displays the matches for each pair.

    Returns:
        tuple: A tuple containing (all_keypoints, all_matches).
            - all_keypoints (list): A list where all_keypoints[i] contains the
                                    cv2.KeyPoint objects for images[i].
            - all_matches (list): A list where all_matches[i] contains the
                                  list of cv2.DMatch objects for the pair
                                  (images[i], images[i+1]).
    """
    if len(images) < 2:
        print("Error: At least two images are required for matching.")
        return [], []

    # 1. Initialize the ORB detector and Brute-Force Matcher
    # ORB is a fast and free alternative to SIFT and SURF.
    orb = cv2.ORB_create(nfeatures=2000)  # Increase features for better matching

    # NORM_HAMMING is used for binary descriptors like ORB.
    # crossCheck=True provides more robust matches by ensuring a two-way match.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 2. Detect keypoints and compute descriptors for all images in a first pass
    all_keypoints = []
    all_descriptors = []
    print("Detecting keypoints and descriptors for all images...")
    for i, img in enumerate(images):
        # Convert to grayscale for feature detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray_img, None)
        all_keypoints.append(kp)
        all_descriptors.append(des)
        print(f"  - Found {len(kp)} keypoints in image {i}.")

    # 3. Match descriptors between consecutive images
    all_matches = []
    print("\nMatching features between consecutive image pairs...")
    for i in range(len(images) - 1):
        print(f"  - Matching image {i} with image {i+1}...")
        
        # Match descriptors from image i and i+1
        matches = bf.match(all_descriptors[i], all_descriptors[i+1])

        # Sort matches by their distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)
        all_matches.append(matches)
        print(f"    -> Found {len(matches)} matches.")

        # 4. Visualize the matches if the flag is set
        if visualize:
            # Draw the top 50 matches for clarity
            num_matches_to_draw = min(len(matches), 50)
            
            match_img = cv2.drawMatches(
                images[i], all_keypoints[i],
                images[i+1], all_keypoints[i+1],
                matches[:num_matches_to_draw], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Resize for better display if images are large
            h, w, _ = match_img.shape
            if w > 1920:
                scale = 1920 / w
                h_new, w_new = int(h * scale), int(w * scale)
                match_img_resized = cv2.resize(match_img, (w_new, h_new))
            else:
                match_img_resized = match_img

            cv2.imshow(f'Matches between Image {i} and {i+1}', match_img_resized)
            print("      -> Displaying visualization. Press any key to continue to the next pair...")
            cv2.waitKey(0)
            cv2.destroyWindow(f'Matches between Image {i} and {i+1}')

    if visualize:
        print("\nVisualization complete.")
        
    return all_keypoints, all_matches


if __name__ == '__main__':
     # --- Main execution block ---
    parser = argparse.ArgumentParser(description='Perform sequential feature matching on images in a folder.')
    parser.add_argument('folder', type=str, help='Path to the folder containing images.')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of matches.')
    args = parser.parse_args()

    image_folder = args.folder
    
    # Check if the folder exists
    if not os.path.isdir(image_folder):
        print(f"Error: Folder not found at '{image_folder}'")
        exit()

    # Find all image files in the folder
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    
    # Sort the images by filename to ensure correct order
    image_paths.sort()

    if not image_paths:
        print(f"Error: No images found in '{image_folder}'")
        exit()

    # Read images from the sorted paths
    image_sequence = [cv2.imread(path) for path in image_paths]
    
    print(f"--- Found and loaded {len(image_sequence)} images from '{image_folder}' ---")
    
    # Run the feature matching function
    keypoints, matches = sequential_feature_matching(image_sequence, visualize=args.visualize)

    print("\n--- Feature Matching Summary ---")
    for i, match_list in enumerate(matches):
        print(f"Total matches found between image {i} ({os.path.basename(image_paths[i])}) and {i+1} ({os.path.basename(image_paths[i+1])}): {len(match_list)}")
    
    print("\nScript finished.")
