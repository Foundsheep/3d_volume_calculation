import open3d as o3d
import open3d.core as o3c
import numpy as np

from .utils import time_check
from .visualise import draw_registration_result
import time

@time_check
def global_register(source, target):
    def preprocess_point_cloud(pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def prepare_dataset(voxel_size):
        print(":: Load two point clouds and disturb initial pose.")

        demo_icp_pcds = o3d.data.DemoICPPointClouds()
        source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        # draw_registration_result(source, target, np.identity(4))

        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh
    
    def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    voxel_size = 0.05
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    # draw_registration_result(source_down, target_down, result_fast.transformation)
    return source_down, target_down