import open3d as o3d
import open3d.core as o3c
# from open3d.web_visualizer import draw


from .utils import time_check

@time_check
def get_hull_and_lines(pcd):
    hull, _ = pcd.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    return hull, hull_ls

@time_check
def get_voxel_grid(pcd, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size)
    return voxel_grid
