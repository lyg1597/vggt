import os 
import shutil 
import json 
import argparse 
import copy 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy and sort images from a folder, skipping every Nth image.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i","--input_folder",
        type=str,
        required=True, 
        help="Path to the folder containing the source images."
    )
    parser.add_argument(
        "-o","--output_folder",
        type=str,
        required=True, 
        help="Path to the folder where the copied images will be saved."
    )
    parser.add_argument(
        "-n", "--skip-interval",
        type=int,
        required=True,
        help="The interval for skipping images (e.g., 3 means skip every 3rd image)."
    )

    args = parser.parse_args()

    input_json_fn = os.path.join(args.input_folder, 'transforms.json')

    with open(input_json_fn,'r') as f:
        transforms_json = json.load(f)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    new_transforms_json = copy.deepcopy(transforms_json)
    new_transforms_json['frames'] = []

    frames = transforms_json['frames']
    for i in range(500,len(frames),args.skip_interval):
        frame = frames[i]
        new_transforms_json['frames'].append(frame)
        
        img_fn = frame['file_path']
        input_image_fn = os.path.join(args.input_folder, img_fn)
        output_image_fn = os.path.join(args.output_folder, img_fn)

        shutil.copy(input_image_fn, output_image_fn)

    output_json_fn = os.path.join(args.output_folder, 'transforms.json')
    with open(output_json_fn, 'w+') as f:
        json.dump(new_transforms_json, f, indent=3)