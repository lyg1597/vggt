import os
import shutil
import argparse
import re

def natural_sort_key(s):
    """
    Create a sort key that handles numbers in filenames naturally.
    e.g., 'image10.jpg' comes after 'image2.jpg'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def copy_images_with_skip(input_folder, output_folder, n):
    """
    Sorts images by filename, copies them to a new folder, and skips every nth image.

    Args:
        input_folder (str): The path to the folder containing the original images.
        output_folder (str): The path where the selected images will be saved.
        n (int): The interval for skipping images. Must be a positive integer.
    """
    # --- 1. Validate inputs and set up folders ---
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    if n <= 0:
        print("Error: The skip interval 'n' must be a positive integer.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # --- 2. Get and filter image files ---
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    try:
        all_files = os.listdir(input_folder)
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    except Exception as e:
        print(f"Error reading from input folder: {e}")
        return

    if not image_files:
        print(f"No images found in '{input_folder}'.")
        return

    # --- 3. Sort the images by filename ---
    image_files.sort(key=natural_sort_key)
    print(f"Found and sorted {len(image_files)} images.")

    # --- 4. Copy images, skipping every nth file ---
    copied_count = 0
    skipped_count = 0

    for i in range(0, len(image_files), n):
        filename = image_files[i]
        # The loop index 'i' is 0-based. We use (i + 1) for 1-based counting.
        # If (i + 1) is a multiple of n, we skip it.
        # if (i + 1) % n == 0:
        #     print(f"Skipping ({i + 1}/{len(image_files)}): {filename}")
        #     skipped_count += 1
        #     continue

        # This is not an nth image, so we copy it.
        print(f"Copying ({i + 1}/{len(image_files)}): {filename}")
        source_path = os.path.join(input_folder, filename)
        destination_path = os.path.join(output_folder, filename)

        try:
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        except Exception as e:
            print(f"Could not copy file {filename}. Error: {e}")

    # --- 5. Print a final summary ---
    print("\n--- Process Complete ---")
    print(f"Total images copied: {copied_count}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Output saved to: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy and sort images from a folder, skipping every Nth image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing the source images."
    )
    parser.add_argument(
        "output_folder",
        help="Path to the folder where the copied images will be saved."
    )
    parser.add_argument(
        "-n", "--skip-interval",
        type=int,
        required=True,
        help="The interval for skipping images (e.g., 3 means skip every 3rd image)."
    )

    args = parser.parse_args()
    copy_images_with_skip(args.input_folder, args.output_folder, args.skip_interval)