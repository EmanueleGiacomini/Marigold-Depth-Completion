import numpy as np
import argparse
import logging
import os
import yaml
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
from cycler import cycle



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize camera trajectories. Used to validate GT trajectories for NeRFStudio")

    parser.add_argument(
        "--input_trajectory",
        type=str,
        required=True,
        help="Input file containing the trajectory of the camera in numpy format"
    )

    parser.add_argument(
        "--camera_calibration",
        type=str,
        required=True,
        help="Input camera calibration file. Must contain fx, fy, cx, cy and distortion coefficients"
    )

    args = parser.parse_args()

    input_trajectory = args.input_trajectory
    input_camera_calibration = args.camera_calibration

    # -------------- Camera Intrinsics --------------#
    # TODO: Accept other formats, for now, YAML should be fine
    with open(input_camera_calibration, "r") as f:
        calibration_dict = yaml.safe_load(f)
        K = np.eye(3, 3, dtype=np.float32)
        K[0, 0] = calibration_dict["K"][0]
        K[1, 1] = calibration_dict["K"][1]
        K[0, 2] = calibration_dict["K"][2]
        K[1, 2] = calibration_dict["K"][3]
        dist_coeffs = np.asarray(calibration_dict["dist_coeffs"])
        width, height = calibration_dict["image_size"]
    print(K)
    print(dist_coeffs)
    print(width, height)

    # ------------- Read Trajectory ---------------#
    trajectory = np.loadtxt(input_trajectory)
    # for t in range(len(trajectory)):
        # print(trajectory[t, 3:])
        # trajectory[t, 3:] = pr.quaternion_wxyz_from_xyzw(trajectory[t, 3:])
    cam2world_trajectory = ptr.transforms_from_pqs(trajectory)

    sensor_size = np.array([0.036, 0.024])
    virtual_image_distance = 1
    intrinsic_matrix = K

    plt.figure(figsize=(5, 5))
    ax = pt.plot_transform(s=0.3)
    ax = ptr.plot_trajectory(ax, P=trajectory, s=0.1, n_frames=10)

    key_frames_indices = np.linspace(0, len(trajectory) - 1, 10, dtype=int)
    colors = cycle("rgb")
    image_size = np.array([width, height])
    for i, c in zip(key_frames_indices, colors):
        pc.plot_camera(ax, intrinsic_matrix, cam2world_trajectory[i],
                       sensor_size=image_size, virtual_image_distance=0.2, c=c)
    # for pose in trajectory:
    #     world_T_cam = pt.transform_from_pq(pose)
    trajectory = np.asarray(trajectory)
    pos_min = np.min(trajectory[:, :3], axis=0)
    pos_max = np.max(trajectory[:, :3], axis=0)
    center = (pos_max + pos_min) / 2.0
    max_half_extent = max(pos_max - pos_min) / 2.0
    ax.set_xlim((center[0] - max_half_extent, center[0] + max_half_extent))
    ax.set_ylim((center[1] - max_half_extent, center[1] + max_half_extent))
    ax.set_zlim((center[2] - max_half_extent, center[2] + max_half_extent))
    plt.show()

    exit(0)
