import open3d as o3d
import open3d.core as o3c
import numpy as np

from .utils import time_check

@time_check
def read_from_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.paint_uniform_color([1, 1, 1])
    return pcd