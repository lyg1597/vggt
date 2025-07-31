import shutil 
import json 
import os 

with open('./sampled_big_room_undistort/transforms.json','r') as f:
    json_data = json.load(f)

frames = json_data['frames']

for frame in frames:
    input_fn = os.path.join('./big_room', frame['file_path'])
    
    output_fn = os.path.join('sampled_big_room/', frame['file_path'])

    shutil.copy(input_fn, output_fn)