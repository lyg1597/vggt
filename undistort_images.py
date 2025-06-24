import os
import re
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def create_transform_matrix(position, quaternion):
    """
    Converts position and quaternion (x, y, z, w) to a 4x4 transformation matrix.

    Args:
        position (np.array): A 3-element array for (x, y, z).
        quaternion (np.array): A 4-element array for (x, y, z, w).

    Returns:
        np.array: A 4x4 transformation matrix.
    """
    # Create a 3x3 rotation matrix from the quaternion
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Create a 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position

    return transform_matrix

def process_files_to_json(input_folder, output_folder):
    """
    Matches odom and rgb files, undistorts images, and generates a JSON
    file with camera poses for NeRF-style datasets.

    Args:
        input_folder (str): The path to the folder containing the odom and rgb files.
        output_folder (str): The path where the processed files will be saved.
    """
    # --- Create the output folder if it doesn't exist ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # --- Camera parameters and JSON structure ---
    json_data = {
        "w": 1296,
        "h": 972,
        "fl_x": 624.7025861923801,
        "fl_y": 622.1024899860136,
        "cx": 655.0802437532551,
        "cy": 494.9500170238977,
        "k1": -0.1633556496674091,
        "k2": 0.019164905599367132,
        "p1": 7.588347139885713e-05,
        "p2": -0.00010445711172148531,
        "camera_model": "OPENCV",
        "frames": []
    }

    camera_matrix = np.array([
        [json_data['fl_x'], 0, json_data['cx']],
        [0, json_data['fl_y'], json_data['cy']],
        [0, 0, 1]
    ])
    dist_coeffs = np.array([
        json_data['k1'], json_data['k2'], json_data['p1'], json_data['p2']
    ])

    # --- Match odom and rgb files ---
    file_pattern = re.compile(r'(odom|rgb)_(\d+\.\d+)\.(txt|jpg)')
    file_map = {}

    for filename in os.listdir(input_folder):
        match = file_pattern.match(filename)
        if match:
            file_type, number, _ = match.groups()
            if number not in file_map:
                file_map[number] = {}
            file_map[number][file_type] = filename

    # Sort the file pairs numerically to ensure frames are in order
    sorted_numbers = sorted(file_map.keys(), key=float)
    colmap_id = 1

    # --- Process matched files ---
    for number in sorted_numbers:
        files = file_map[number]
        if 'odom' in files and 'rgb' in files:
            odom_filename = files['odom']
            rgb_filename = files['rgb']

            odom_filepath = os.path.join(input_folder, odom_filename)
            rgb_filepath = os.path.join(input_folder, rgb_filename)

            # Skip if odom file is empty
            if os.path.getsize(odom_filepath) == 0:
                print(f"Skipping empty odom file: {odom_filename}")
                continue

            print(f"Processing pair: {odom_filename}, {rgb_filename}")

            try:
                # --- Read odom data (position and orientation) ---
                with open(odom_filepath, 'r') as f:
                    pose_data = list(map(float, f.read().strip().split()))
                
                if len(pose_data) != 7:
                    print(f"Warning: Odom file {odom_filename} does not contain 7 elements. Skipping.")
                    continue

                position = np.array(pose_data[:3])
                quaternion = np.array(pose_data[3:]) # Expects x, y, z, w

                # --- Undistort the image ---
                img = cv2.imread(rgb_filepath)
                if img is None:
                    print(f"Warning: Could not read image {rgb_filename}. Skipping.")
                    continue
                
                undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

                # --- Save the undistorted image ---
                output_rgb_filename = f"rgb_{number}.jpg"
                output_rgb_filepath = os.path.join(output_folder, output_rgb_filename)
                cv2.imwrite(output_rgb_filepath, undistorted_img)
                
                # --- Create frame object for JSON ---
                transform_matrix = create_transform_matrix(position, quaternion)
                
                frame_data = {
                    "file_path": output_rgb_filename, # Relative path
                    "transform_matrix": transform_matrix.tolist(),
                    "colmap_im_id": colmap_id
                }

                json_data["frames"].append(frame_data)
                colmap_id += 1

            except Exception as e:
                print(f"An error occurred while processing {rgb_filename}: {e}")

    # --- Write the final JSON file ---
    json_output_path = os.path.join(output_folder, "transforms.json")
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\nProcessing complete. JSON file saved to: {json_output_path}")

if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Replace these paths with your actual input and output folder paths
    input_directory = "big_room"
    output_directory = "big_room_undistort"

    if input_directory == "path/to/your/input_folder" or output_directory == "path/to/your/output_folder":
        print("Please update the 'input_directory' and 'output_directory' variables in the script.")
    else:
        process_files_to_json(input_directory, output_directory)