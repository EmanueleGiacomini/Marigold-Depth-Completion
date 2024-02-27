import argparse
import logging
import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

EXTENSION_LIST = [".tiff", ".png", ".npy"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare data for KITTI Dev-Completion benchmark"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the depth-maps to benchmark"
    )

    parser.add_argument(
        "--lidar_dir",
        type=str,
        required=True,
        help="Path to LiDAR projections. Used to compute the absolute scale"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to write results in the proper format"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    lidar_dir = args.lidar_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    input_filename_list = glob(os.path.join(input_dir, "*"))
    input_filename_list = [
        f for f in input_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]

    input_filename_list = sorted(input_filename_list)
    if len(input_filename_list) > 0:
        logging.info(f"Found {len(input_filename_list)} images")
    else:
        logging.error(f"No image found in '{input_dir}'")
        exit(1)

    lidar_filename_list = glob(os.path.join(lidar_dir, "*"))
    lidar_filename_list = [
        f for f in lidar_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    lidar_filename_list = sorted(lidar_filename_list)

    if len(lidar_filename_list) > 0:
        logging.info(f"Found {len(lidar_filename_list)} GT Maps")
    else:
        logging.error(f"No GT Maps found in '{lidar_dir}")
        exit(1)
    # Associate files

    matching_pairs = [
        (est_f, lidar_f)
        for est_f in input_filename_list
        for lidar_f in lidar_filename_list
        if os.path.splitext(os.path.basename(est_f))[0][:-5] ==
           os.path.splitext(os.path.basename(lidar_f))[0]
    ]

    num_pairs = len(matching_pairs)
    if num_pairs > 0:
        logging.info(f"Found {num_pairs} matching pairs")
    else:
        logging.error(f"Found no matching pairs between input sets")
        exit(1)

    for input_filename, lidar_filename in tqdm(matching_pairs):
        if input_filename.endswith(".npy"):
            input_image = np.load(input_filename)
        else:
            input_image = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)

        if lidar_filename.endswith(".npy"):
            lidar_image = np.load(lidar_filename)
        else:
            lidar_image = cv2.imread(lidar_filename, cv2.IMREAD_UNCHANGED)

        mask = lidar_image > 0

        valid_gt = lidar_image[mask]
        valid_depths = input_image[mask]

        A = np.vstack((valid_depths, np.ones_like(valid_depths))).T
        a, b = np.linalg.lstsq(A, valid_gt, rcond=-1)[0]
        input_image = (input_image * a + b).astype(np.uint16)

        output_filename = os.path.splitext(os.path.basename(lidar_filename))[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, output_filename), input_image)

    exit(0)
