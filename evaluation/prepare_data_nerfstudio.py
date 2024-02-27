import argparse
import logging
from glob import glob
from itertools import accumulate

from tqdm import tqdm
import numpy as np
import os
import shutil
import yaml
import json
from pytransform3d.transformations import (
    transform_from_pq,
    pq_from_transform
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a [VBR/KITTI] sequence, prepare data to be ran on Nerfstudio')

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help="Path to directory of images to process"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Output directory that is "
    )

    parser.add_argument(
        '--input_gt',
        type=str,
        required=True
    )

    parser.add_argument(
        '--from_index',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--to_index',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['kitti', 'vbr'],
        required=True
    )

    parser.add_argument(
        "--transform_gt_in_camera",
        type=str,
        required=False,
        help="Input file containing a rigid-body transform that maps GT coordinates in camera coordinates"
    )

    parser.add_argument(
        "--skip_gt_rows",
        type=int,
        required=False,
        help="No. of rows to skip during the parsing of GT",
        default=0
    )

    parser.add_argument(
        "--quaternion_format",
        type=str,
        required=False,
        choices=["wxyz", "xyzw"],
        default="xyzw",
        help="Defines the format of quaternions for handling GT coordinates"
    )

    parser.add_argument(
        "--camera_calibration",
        type=str,
        required=True,
        help="Input camera calibration file. Must contain fx, fy, cx, cy and distortion coefficients"
    )

    parser.add_argument(
        "--save_gt",
        type=str,
        required=False,
        help="If set, saves the GT trajectory prior to OpenGL format conversion"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        help="prefix to images after the conversion in nerfstudio format",
        default=""
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    input_gt = args.input_gt
    from_idx = args.from_index
    to_idx = args.to_index
    dataset_format = args.dataset
    skip_gt_rows = args.skip_gt_rows
    input_camera_calibration = args.camera_calibration
    input_transform_gt_in_camera = args.transform_gt_in_camera

    output_dir_images = os.path.join(output_dir, 'images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # Get path to all images
    rgb_filename_list = glob(os.path.join(input_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1] in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)

    # Temporary solution. We don't know how the fuck to do this stuff
    if from_idx != -1 or to_idx != -1:
        raise RuntimeError("from_index and to_index functionalities are not yet implemented :(")

    if from_idx == -1:
        from_idx = os.path.basename(rgb_filename_list[0])
    if to_idx == -1:
        to_idx = os.path.basename(rgb_filename_list[-1])

    rgb_filename_list = list(filter(lambda x: from_idx <= os.path.basename(x) <= to_idx, rgb_filename_list))

    # -------------- Ground Truth --------------#
    if dataset_format == "kitti":
        from_idx_n = int(os.path.splitext(from_idx)[0])
        to_idx_n = int(os.path.splitext(to_idx)[0])
        with open(input_gt, "r") as f:
            lines = f.readlines()[skip_gt_rows:]
            gt_list = list(
                map(lambda x: np.vstack((np.fromstring(x.strip(), sep=" ").reshape((3, 4)),
                                         np.float32([0., 0., 0., 1]))),
                    lines)
            )
    elif dataset_format == "vbr":
        with open(input_gt, "r") as f:
            lines = f.readlines()[skip_gt_rows:]
            gt_list = list(
                map(lambda x: np.fromstring(x.strip(), sep=" ")[1:], lines)
            )
            if args.quaternion_format == "xyzw":
                gt_list = list(
                    map(lambda x: transform_from_pq(np.float32([x[0], x[1], x[2], x[6], x[3], x[4], x[5]])), gt_list)
                )
            else:
                gt_list = list(
                    map(lambda x: transform_from_pq(x), gt_list)
                )

    # ------- Apply Transform to express GT in camera frame ------
    if input_transform_gt_in_camera is not None:
        with open(input_transform_gt_in_camera, "r") as f:
            ctg_dict = yaml.safe_load(f)
            camera_T_gt = np.eye(4, 4, dtype=np.float32)
            camera_T_gt[:3, :3] = np.asarray(ctg_dict["R"], dtype=np.float32).reshape(3, 3)
            camera_T_gt[:3, 3] = np.asarray(ctg_dict["t"], dtype=np.float32)
        logging.info("Computing GT with respect to camera")
        logging.info(f"camera_T_gt = {pq_from_transform(camera_T_gt)}")

        gt_T_camera = np.linalg.inv(camera_T_gt)
        gt_list = [x @ gt_T_camera for x in gt_list]

        # gt_traj = np.array(gt_list)
        # inv_gt_traj = np.linalg.inv(gt_traj)
        # relative_poses = np.matmul(inv_gt_traj[:-1], gt_traj[1:])
        # relative_poses = np.split(relative_poses, relative_poses.shape[0])
        # relative_poses = [np.eye(4, 4, dtype=np.float32)] + relative_poses
        # relative_poses = [camera_T_gt] + relative_poses
        # relative_poses_cam = [camera_T_gt @ x.squeeze() @ np.linalg.inv(camera_T_gt) for x in relative_poses]
        # relative_poses_cam = [np.linalg.inv(camera_T_gt) @ x.squeeze() @ camera_T_gt for x in relative_poses]
        # relative_poses_cam = [np.eye(4, 4, dtype=np.float32)] + relative_poses_cam
        # gt_list_prev = gt_list
        # gt_list = list(accumulate(relative_poses_cam, np.matmul))
        # Temp
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # gt = np.array(gt_list)
        # gt_prev = np.array(gt_list_prev)
        # gt_gl = gt
        # gt_gl[:, 1, :] *= -1
        # gt_gl[:, 2, :] *= -1
        #
        # ax.plot(gt[:, 0, 3], gt[:, 2, 3], gt[:, 1, 3])
        # ax.plot(gt_prev[:, 0, 3], gt_prev[:, 1, 3], gt_prev[:, 2, 3])
        # ax.plot(gt_gl[:, 0, 3], gt_gl[:, 2, 3], gt_gl[:, 1, 3])
        # plt.show()
        # exit(0)

    # ------ Save GT -------
    if args.save_gt:
        logging.info(f"Saving GT trajectory to {args.save_gt}")
        pq_list = list(map(pq_from_transform, gt_list))
        pq_list = np.asarray(pq_list)
        np.savetxt(args.save_gt, pq_list)

    print(len(rgb_filename_list))
    print(len(gt_list))
    logging.info("Assuming GT and rgb images starts from the same index")
    rgb_filename_list = rgb_filename_list[:len(gt_list)]

    # -------------- Camera Intrinsics --------------#
    # TODO: Accept other formats, for now, YAML should be fine
    with open(input_camera_calibration, "r") as f:
        calibration_dict = yaml.safe_load(f)

    fx, fy, cx, cy = calibration_dict["K"]
    k1, k2, p1, p2, _ = calibration_dict["dist_coeffs"]

    width, height = calibration_dict["image_size"]
    print(f" fx={fx} fy={fy} cx={cx} cy={cy}")
    print(f" k1={k1} k2={k2} p1={p1} p2={p2}")
    print(f" image_width={width} image_height={height}")

    # Generate transforms.json file

    # transforms_dict = {"fl_x": fx, "fl_y": fy, "k1": k1, "k2": k2, "p1": p1, "p2": p2, "cx": cx, "cy": cy, "w": width,
    #                    "h": height, "frames": list()}
    transforms_dict = {"camera_model": "OPENCV", "frames": list()}

    for T_cam, rgb_path in tqdm(zip(gt_list, rgb_filename_list), total=len(gt_list)):
        # inverse_matrix = np.array(
        #     [[1.0, 1.0, 1.0, 1.0],
        #      [-1.0, -1.0, -1.0, -1.0],
        #      [-1.0, -1.0, -1.0, -1.0],
        #      [1.0, 1.0, 1.0, 1.0]])
        # T_cam = T_cam * inverse_matrix

        # Magic shit happening down here!
        # Taken by https://github.com/nerfstudio-project/nerfstudio/blob/c256dc2bf95e2ef8d2dcc13403c551700e031df4/nerfstudio/process_data/colmap_utils.py#L388
        T_cam[0:3, 1:3] *= -1
        T_cam = T_cam[np.array([1, 0, 2, 3]), :]
        T_cam[2, :] *= -1
        frame = {
            "file_path": os.path.join("./images", args.prefix + os.path.basename(rgb_path)),
            "fl_x": fx,
            "fl_y": fy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            # "k1": k1,
            # "k2": k2,
            # "p1": p1,
            # "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": width,
            "h": height,
            "transform_matrix": [
                T_cam[0, :].tolist(),
                T_cam[1, :].tolist(),  # -1 * T_cam[{1,2}, :] to pass in OpenGL format
                T_cam[2, :].tolist(),
                T_cam[3, :].tolist()
            ]
        }
        transforms_dict["frames"].append(frame)
        # Copy images
        shutil.copy(rgb_path,
                    os.path.join(
                        output_dir_images,
                        args.prefix + os.path.basename(rgb_path)
                    ))

    with open(os.path.join(output_dir, "transforms.json"), "w") as f:
        json.dump(transforms_dict, f)

    exit(0)
