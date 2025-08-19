import os 
import shutil 
import json 
import argparse 
import copy 
import numpy as np 

if __name__ == "__main__":
    folder = './big_room_undistort_rename'
    output_folder = './dist_sampled_big_room_undistort_0300'

    json_fn = os.path.join(folder, './transforms.json')

    with open(json_fn, 'r') as f:
        json_data = json.load(f)

    new_json_data = copy.deepcopy(json_data)

    frames = json_data['frames']

    dist = float('inf')
    prev_pos = np.array([0,0,0])
    
    new_frames = []
    new_idx = 0
    for i in range(len(frames)):
        frame = frames[i]    
        transform_matrix = np.array(frame['transform_matrix'])

        image_fn = frame['file_path']
        image_path = os.path.join(folder, image_fn)

        if not os.path.exists(image_path):
            continue 
        
        current_rotation = transform_matrix[:3,:3]
        current_position = transform_matrix[:3,3]
        dist = dist + np.linalg.norm(prev_pos - current_position)

        if dist > 0.3:
            new_frame = copy.deepcopy(frame)
            new_frame['file_path'] = f'frame_{new_idx:05d}.png'
            new_frame['prev_file_path'] = image_fn
            new_frames.append(new_frame)
            shutil.copy(image_path, os.path.join(output_folder, f'frame_{new_idx:05d}.png'))
            new_idx += 1
            dist = 0
        prev_pos = current_position

    new_json_data['frames'] = new_frames

    with open(os.path.join(output_folder, 'transforms.json'), 'w+') as f:
        json.dump(new_json_data, f, indent=4)
 
