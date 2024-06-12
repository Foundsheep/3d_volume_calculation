import open3d as o3d
import open3d.core as o3c
# from open3d.web_visualizer import draw

import matplotlib.pyplot as plt
import h5py
import numpy as np

import time
import datetime

from src import file_io, preprocess, registration, visualise, calculate_volume





def evaluate(source, target, threshold, trans_init):
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)


def register(source, target, threshold, trans_init):
    start = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print(f"=== time took : [{time.time() - start :.2f}]")
    return source, target, reg_p2p.transformation


    



if __name__ == "__main__":

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    total_time = 0.0
    path_1 = "./datasets/angle_3.ply"

    # --- read file
    pcd_1, temp_time = file_io.read_from_ply(path_1)
    total_time += temp_time

    # --- pre-processing
    pcd_1, temp_time = preprocess.down_sample_pcd(pcd_1, voxel_size=10)
    total_time += temp_time
    visualise.visualise([pcd_1])
    visualise.capture_visuals([pcd_1], "down_sample_pcd", timestamp)

    # # --- rotate
    # pcd_test, temp_time = visualise.rotate_pcd(pcd_1)
    # total_time += temp_time
    # visualise.visualise([pcd_1, pcd_test])

    (inlier_cloud, outlier_cloud), temp_time = preprocess.remove_outlier(pcd_1)
    total_time += temp_time
    visualise.visualise([inlier_cloud, outlier_cloud])
    visualise.capture_visuals([inlier_cloud, outlier_cloud], "remove_outlier", timestamp)

    (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    total_time += temp_time
    visualise.visualise([inlier_cloud, outlier_cloud])
    visualise.capture_visuals([inlier_cloud, outlier_cloud], "segment_plane_1", timestamp)

    (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(outlier_cloud)
    total_time += temp_time
    visualise.visualise([outlier_cloud])
    visualise.capture_visuals([outlier_cloud], "segment_plane_2", timestamp)

    (inlier_cloud, outlier_cloud), temp_time = preprocess.remove_outlier(outlier_cloud)
    total_time += temp_time
    visualise.visualise([inlier_cloud])
    visualise.capture_visuals([inlier_cloud], "remove_outlier", timestamp)

    (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    total_time += temp_time
    visualise.visualise([outlier_cloud])
    visualise.capture_visuals([outlier_cloud], "segment_plane_3", timestamp)


    # --- volume calculation
    (hull, hull_ls), temp_time = calculate_volume.get_hull_and_lines(outlier_cloud)
    total_time += temp_time
    visualise.visualise([outlier_cloud, hull_ls])
    visualise.capture_visuals([outlier_cloud, hull_ls], "get_hull_and_lines", timestamp)
    print(f"volume: [{hull.get_volume()}]")
    print(f"total time: [{total_time :.6f}] seconds")








    # global_register(pcd_1, pcd_2)
    # get_dummy_crop_data(pcd_1)
    # source, target, threshold, trans_init = get_dummy_icp_data()
    # evaluate(source, target, threshold, trans_init)
    # source, target, trans = register(source, target, threshold, trans_init)
    # draw_registration_result(source, target, trans)
