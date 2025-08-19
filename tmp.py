import shutil 
import json 
import os 

with open('./big_room_undistort_rename/transforms.json','r') as f:
    json_data = json.load(f)

frames = json_data['frames']
new_frames = []
for i in range(len(frames)):
    image_fn = f'./big_room_undistort_rename/frame_{i:05d}.jpg'
    if not os.path.exists(image_fn):
        continue 
    new_frame = frames[i]
    new_frame['file_path'] = f'frame_{i:05d}.jpg'
    new_frames.append(new_frame)

json_data['frames'] = new_frames 
with open('./big_room_undistort_rename/transforms.json','r+') as f:
    json.dump(json_data, f, indent=4)