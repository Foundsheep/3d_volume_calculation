import open3d as o3d
import open3d.core as o3c
import numpy as np
import matplotlib.pyplot as plt

from .utils import time_check

@time_check
def down_sample_pcd(pcd, voxel_size):
    print(f"=== original : {pcd}")
    down_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"=== down_pcd : {down_pcd}")
    return down_pcd

@time_check
def segment_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=2,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    outlier_cloud.paint_uniform_color([1.0, 0, 0])
    return inlier_cloud, outlier_cloud

@time_check
def remove_outlier(pcd):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
                                             std_ratio=0.5)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    outlier_cloud.paint_uniform_color([1.0, 0, 0])
    return inlier_cloud, outlier_cloud


def dbscan_clustring(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.5, min_points=100, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd