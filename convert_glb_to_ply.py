import trimesh
import numpy as np

# Define the input and output file paths
input_glb_path = 'glbscene_50_All_maskbFalse_maskwFalse_camFalse_skyFalse_predDepthmap_and_Camera_Branch.glb'
output_ply_path = 'output_point_cloud.ply'

# --- Step 1: Load the GLB file as a scene ---
print(f"Loading scene from: {input_glb_path}")
scene = trimesh.load(input_glb_path)

# --- Step 2: Iterate through geometries and find the PointCloud ---
point_cloud = None
print("Inspecting geometries in the scene...")
for geom in scene.geometry.values():
    print(f"  - Found geometry of type: {type(geom)}")
    # We are looking for the specific PointCloud object
    if isinstance(geom, trimesh.points.PointCloud):
        point_cloud = geom
        print(f"    -> Identified as the PointCloud to be extracted.")
        break # Stop after finding the first one

# --- Step 3: Check if a PointCloud was actually found ---
if point_cloud is None:
    raise ValueError("No trimesh.PointCloud object was found in the GLB scene.")

if len(point_cloud.vertices) == 0:
    raise ValueError("The PointCloud object in the scene contains 0 vertices.")

print(f"Successfully extracted a point cloud with {len(point_cloud.vertices)} points.")

# --- Step 4: Export the extracted PointCloud to a PLY file ---
# We can directly use the export method of the PointCloud object
point_cloud.export(output_ply_path, file_type='ply', encoding='ascii')

print(f"Successfully converted and saved the point cloud to {output_ply_path}")

