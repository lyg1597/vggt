import cv2
import numpy as np
from pathlib import Path
import re
from PIL import Image
import os

def select_keyframes(images, N, threshold=75, visualize = False, output_folder=None):
    """
    Selects keyframes from a sequence of images based on feature matching.

    This function identifies frames that are significantly different from the previous
    keyframe, ensuring that the selected frames provide new information for tasks
    like 3D reconstruction.

    Args:
        images (np.ndarray): A NumPy array of shape (num_frames, H, W, 3) containing
                             the image sequence in OpenCV's default BGR format.
        N (int): The maximum number of keyframes to return.
        threshold (int): The minimum number of "good" feature matches required
                         to consider a frame redundant. A new keyframe is chosen when
                         the number of matches drops below this value.

    Returns:
        np.ndarray: An array of shape (M, H, W, 3) containing the selected keyframes,
                    where M <= N.
    """
    # if not isinstance(images, np.ndarray) or images.ndim != 4:
    #     raise ValueError("Input 'images' must be a 4D NumPy array (N, H, W, 3).")
    # if not images.any() or N == 0:
    #     return np.array([])

    # Initialize ORB feature detector and Brute-Force matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    keyframes = []
    keyframe_indices = []

    # The first frame is always a keyframe
    with Image.open(images[0]) as img:
        first_frame = np.array(img)
    keyframes.append(first_frame)
    keyframe_indices.append(0)

    # Compute features for the first keyframe
    gray_last_kf = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    kp_last_kf, des_last_kf = orb.detectAndCompute(gray_last_kf, None)

    for i in range(1, len(images)):
        print(f"process frame{i}")
        # if len(keyframes) >= N:
        #     break  # Stop if we have reached the desired number of keyframes

        with Image.open(images[i]) as img:
            current_frame = np.array(img)
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        kp_current, des_current = orb.detectAndCompute(gray_current, None)

        # Skip frame if no features are found in either image
        if des_last_kf is None or des_current is None:
            continue

        # Match descriptors using k-Nearest Neighbor search
        matches = bf.knnMatch(des_last_kf, des_current, k=2)

        # Apply Lowe's ratio test to filter for good matches
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            # This can happen if knnMatch returns fewer than k=2 matches
            continue

        # Decision: If matches are below the threshold, it's a new keyframe
        if len(good_matches) < threshold:
            # --- Visualization Block ---
            if visualize:
                print(f"✨ New keyframe! Visualizing match between frame {keyframe_indices[-1]} and {i}")
                last_kf_image = keyframes[-1]

                # Draw the matches
                match_img = cv2.drawMatches(
                    last_kf_image, kp_last_kf,
                    current_frame, kp_current,
                    good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # Display the image and wait for user input
                cv2.imshow(f'Match: Frame {keyframe_indices[-1]} vs Frame {i}', match_img)
                print("-> Press any key on the image window to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow(f'Match: Frame {keyframe_indices[-1]} vs Frame {i}')
            
            keyframes.append(current_frame)
            keyframe_indices.append(i)
            
            # Update the last keyframe and its features
            kp_last_kf, des_last_kf = kp_current, des_current

    # --- Added: Save keyframes to disk ---
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
        print(f"💾 Saving {len(keyframes)} keyframes to '{output_folder}'...")
        for idx, frame in enumerate(keyframes):
            # Create a filename with zero-padding (e.g., keyframe_001.png)
            filename = f"keyframe_{idx+1:03d}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"✅ Selected {len(keyframes)} keyframes at indices: {keyframe_indices}")
    return np.array(keyframes)

def reorder_and_rename_images(directory_path):
    """
    Finds images named 'rgb_{%d}.{%d}.jpg', sorts them numerically,
    and saves them as 'frame_{idx:05d}.png'.

    Args:
        directory_path (str): The path to the folder containing the images.
    """
    image_dir = Path(directory_path)
    # new_image_dir = Path(new_image_dir)
    if not image_dir.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
        return

    # 1. Find all files matching the initial pattern
    jpg_files = list(image_dir.glob('rgb_*.jpg'))

    # 2. Define a key for natural sorting (e.g., so 'rgb_2' comes before 'rgb_10')
    def natural_sort_key(filename):
        # Extracts all numbers from the filename and returns them as a list of integers
        return [int(c) for c in re.findall(r'\d+', filename.name)]

    # 3. Sort the files using the natural sort key
    sorted_files = sorted(jpg_files, key=natural_sort_key)
    
    if not sorted_files:
        print("No matching 'rgb_*.jpg' files were found.")
        return

    # 4. Iterate, convert, and rename
    print(f"Found {len(sorted_files)} images to process...")
    res = []
    for idx, old_path in enumerate(sorted_files, start=1):
        res.append(old_path)
    return res
    #     try:
    #         # # Define the new filename, zero-padded to 5 digits
    #         # new_name = f"frame_{idx:05d}.png"
    #         # new_path = new_image_dir / new_name

    #         # Open the original JPG and save it as a new PNG
    #         with Image.open(old_path) as img:
    #             res.append(np.array(img))

    #         # print(f"Converted '{old_path.name}' -> '{new_path.name}'")
            
    #         # --- Optional: Uncomment the line below to delete the original JPG file ---
    #         # old_path.unlink()

    #     except Exception as e:
    #         print(f"Could not process {old_path.name}. Error: {e}")
            
    # # print("\nProcessing complete! ✨")
    # return np.array(res)

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy image sequence for demonstration
    # In a real scenario, you would load your images here
    # num_frames = 100
    # height, width = 480, 640
    
    # # Create a sequence where frames slowly drift
    # dummy_images = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    # for i in range(num_frames):
    #     # Create a simple moving rectangle
    #     start_point = (50 + i * 4, 50)
    #     end_point = (150 + i * 4, 150)
    #     cv2.rectangle(dummy_images[i], start_point, end_point, (0, 255, 0), -1)
    #     cv2.putText(dummy_images[i], f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    images_fn = reorder_and_rename_images('./big_room_undistort')

    print(f"\nInput shape: {len(images_fn)}")

    # Select up to 10 keyframes from the sequence
    max_keyframes_to_select = 100
    selected_kfs = select_keyframes(images_fn, max_keyframes_to_select, threshold=45, visualize=False, output_folder='sampled_big_room_undistort_2')

    print(f"Output shape: {selected_kfs.shape}")

    # To display the keyframes (requires a GUI environment)
    # for i, frame in enumerate(selected_kfs):
    #     cv2.imshow(f'Keyframe {i+1}', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()