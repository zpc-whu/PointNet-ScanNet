import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))

import numpy as np
import pc_util

data = np.load('../npydata_hiRes/scene0001_01.npy')
scene_points = data[:,0:3]
colors = data[:,3:6]
instance_labels = data[:,6]
semantic_labels = data[:,7]


output_folder = 'demo_output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# print str(scene_points)
f = open(os.path.join(output_folder, 'scene.txt'), 'w')
for p in data:
    f.write(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + " " + str(p[6]) + " " + str(p[7]) + "\n")
f.close()
# Write scene as OBJ file for visualization
# pc_util.write_ply(scene_points, os.path.join(output_folder, 'scene.obj'))
# pc_util.write_ply_rgb(scene_points, colors, os.path.join(output_folder, 'scene.obj'))
# pc_util.write_ply_color(scene_points, instance_labels, os.path.join(output_folder, 'scene_instance.obj'))
# pc_util.write_ply_color(scene_points, semantic_labels, os.path.join(output_folder, 'scene_semantic.obj'))
