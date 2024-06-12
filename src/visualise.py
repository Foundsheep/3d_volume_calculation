import open3d as o3d
import open3d.core as o3c
# from open3d.web_visualizer import draw

import copy
import datetime
from pathlib import Path

import numpy as np
from .utils import time_check

def visualise(vis_list):
    o3d.visualization.draw_geometries(vis_list)

    
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
def capture_visuals(vis_list, filename, timestamp):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for geo in vis_list:
        vis.add_geometry(geo)
        vis.update_geometry(geo)
    vis.poll_events()
    vis.update_renderer()

    p = Path(__file__).parent.parent / Path(f"./capture/{timestamp}")
    if not p.exists():
        p.mkdir(parents=True)
        print(f"{str(p)} made!")

    vis.capture_screen_image(f'{str(p)}/{filename}_{timestamp}.png',do_render=True)
    vis.destroy_window()


@time_check
def rotate_pcd(pcd):
    pcd_test = copy.deepcopy(pcd)
    R = pcd_test.get_rotation_matrix_from_xyz((np.pi / 1.5,0, 0))
    pcd_test.rotate(R, center=pcd_test.get_center())
    pcd_test.paint_uniform_color((1, 0, 0))
    return pcd_test
