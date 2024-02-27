import os
from os import path
import yaml
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Outputs the set of transforms needed to run nerfacto and other stuff")

    parser.add_argument(
        "--input_cam_to_cam",
        type=str,
        required=True,
        help="Input cam_to_cam file"
    )

    parser.add_argument(
        "--input_velo_to_cam",
        type=str,
        required=True,
        help="Input velo_to_cam file"
    )

    args = parser.parse_args()

    input_cam_to_cam = args.input_cam_to_cam
    input_velo_to_cam = args.input_velo_to_cam

    # We need to obtain
    # - camera 2 in camera 0
    # - camera 3 in camera 0
    # - velo in camera 2
    # - velo in camera 3

    # ----------- Parse cam to cam -------------

    # ----------- Parse velo to cam ------------


    exit(0)