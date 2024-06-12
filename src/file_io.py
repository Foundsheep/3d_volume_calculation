import open3d as o3d
import open3d.core as o3c
import numpy as np

from .utils import time_check

@time_check
def read_from_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return pcd