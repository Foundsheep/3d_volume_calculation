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

    voxel_size = 5
    total_time = 0.0
    segment_threshold = 400
    time_list = []

    # ---------------- point registration ---------------- 
    # path_1 = "./datasets/left_bottom.ply"
    # path_2 = "./datasets/left_top.ply"
    # path_3 = "./datasets/right_bottom.ply"
    # path_4 = "./datasets/right_top.ply"

    # # --- read file
    # pcd_1, temp_time = file_io.read_from_ply(path_1)
    # total_time += temp_time
    # pcd_2, temp_time = file_io.read_from_ply(path_2)
    # total_time += temp_time
    # pcd_3, temp_time = file_io.read_from_ply(path_3)
    # total_time += temp_time
    # pcd_4, temp_time = file_io.read_from_ply(path_4)
    # total_time += temp_time

    # visualise.visualise([pcd_1, pcd_2, pcd_3, pcd_4])
    # # --- pre-processing
    # pcd_1_down, temp_time = preprocess.down_sample_pcd(pcd_1, voxel_size=voxel_size)
    # total_time += temp_time
    # pcd_2_down, temp_time = preprocess.down_sample_pcd(pcd_2, voxel_size=voxel_size)
    # total_time += temp_time
    # pcd_3_down, temp_time = preprocess.down_sample_pcd(pcd_3, voxel_size=voxel_size)
    # total_time += temp_time
    # pcd_4_down, temp_time = preprocess.down_sample_pcd(pcd_4, voxel_size=voxel_size)
    # total_time += temp_time

    # # TODO: calibrate
    # pcd_combined, temp_time = registration.multiway_register([pcd_1_down, pcd_2_down, pcd_3_down, pcd_4_down], voxel_size)
    # total_time += temp_time
    # visualise.visualise([pcd_combined])
    # visualise.capture_visuals([pcd_combined], "multiway_register", timestamp)
    # ---------------- point registration ---------------- 

    # --- read file    
    print("\n\n::::::::::::::")
    path_1 = "./datasets/angle_3.ply"
    pcd_1, temp_time = file_io.read_from_ply(path_1)
    time_list.append(round(temp_time, 6))
    total_time += temp_time

    # --- rotate
    pcd_rotated, temp_time = visualise.rotate_pcd(pcd_1)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([pcd_rotated])
    visualise.capture_visuals([pcd_rotated], "01_rotate_pcd", timestamp)

    # --- pre-processing
    print("\n\n::::::::::::::")
    pcd_1_down, temp_time = preprocess.down_sample_pcd(pcd_rotated, voxel_size=voxel_size)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([pcd_1_down])
    visualise.capture_visuals([pcd_1_down], "02_01_down_sample_pcd", timestamp)
    
    #     outlier removal
    (inlier_cloud, outlier_cloud), temp_time = preprocess.remove_outlier(pcd_1_down)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud, outlier_cloud])
    visualise.capture_visuals([inlier_cloud, outlier_cloud], "02_02_remove_outlier", timestamp)

    # --- extract object
    print("\n\n::::::::::::::")
    (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud, outlier_cloud])
    visualise.capture_visuals([inlier_cloud, outlier_cloud], "03_01_segment_plane", timestamp)

    # ############################
    # i = 0
    # (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    # total_time += temp_time
    # visualise.visualise([inlier_cloud, outlier_cloud])
    # visualise.capture_visuals([inlier_cloud, outlier_cloud], f"03_01_segment_plane_{str(i).zfill(3)}", timestamp)
    # while len(outlier_cloud.points) > segment_threshold:
    #     i += 1
    #     (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    #     total_time += temp_time
    #     visualise.capture_visuals([inlier_cloud, outlier_cloud], f"03_01_segment_plane_{str(i).zfill(3)}", timestamp)
    # visualise.visualise([inlier_cloud, outlier_cloud])
    # ############################

    inlier_cloud, temp_time = preprocess.dbscan_clustring(inlier_cloud)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud])
    visualise.capture_visuals([inlier_cloud], "03_02_dbscan", timestamp)

    (inlier_cloud, outlier_cloud), temp_time = preprocess.segment_plane(inlier_cloud)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud, outlier_cloud])
    visualise.capture_visuals([inlier_cloud, outlier_cloud], "03_03_segment_plane", timestamp)

    # --- post-processing
    print("\n\n::::::::::::::")
    (inlier_cloud, outlier_cloud), temp_time = preprocess.remove_outlier(inlier_cloud)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud])
    visualise.capture_visuals([inlier_cloud], "04_remove_outlier", timestamp)
    


    # --- volume calculation
    print("\n\n::::::::::::::")
    (hull, hull_ls), temp_time = calculate_volume.get_hull_and_lines(inlier_cloud)
    time_list.append(round(temp_time, 6))
    total_time += temp_time
    visualise.visualise([inlier_cloud, hull_ls])
    visualise.capture_visuals([inlier_cloud, hull_ls], "05_get_hull_and_lines", timestamp)
    print(f"volume: [{hull.get_volume()}]")
    print(f"total time: [{total_time :.6f}] seconds")
    print(f"time: {time_list}")
