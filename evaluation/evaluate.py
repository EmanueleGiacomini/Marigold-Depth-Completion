import argparse
import logging
import os
from glob import glob

EXTENSION_LIST = [".tiff", ".png"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate the L1 loss between predicted depth-maps and ground truth depth-maps."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input depth-maps folder")

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Path to the Ground-Truth depth-maps folder")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--gt_scale",
        type=float,
        required=True,
        help="Inverse scaling factor for Ground-Truth depth-maps "
             "from UINT16 to meters (i.e., meters = pixels / scale)"
    )

    parser.add_argument(
        "--color_map",
        type=str,
        default="Viridis",
        help="Colormap used to render error maps"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    gt_dir = args.gt_dir
    gt_scale = args.gt_scale
    output_dir = args.output_dir

    # Output Directories
    output_dir_error = os.path.join(output_dir, "error_l1")
    output_dir_abs = os.path.join(output_dir, "depth_absolute")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_error, exist_ok=True)
    os.makedirs(output_dir_abs, exist_ok=True)
    logging.info(f"output dir = {output_dir}")


    # Data
    input_filename_list = glob(os.path.join(input_dir, "*"))
    input_filename_list = [
        f for f in input_filename_list if os.path.splitext(f)[1].lower() == EXTENSION_LIST
    ]
    input_filename_list = sorted(input_filename_list)
    if len(input_filename_list) > 0:
        logging.info(f"Found {len(input_filename_list)} images")
    else:
        logging.error(f"No image found in '{input_dir}'")
        exit(1)

    gt_filename_list = glob(os.path.join(gt_dir, "*"))
    gt_filename_list = [
        f for f in gt_filename_list if os.path.splitext(f)[1].lower() == EXTENSION_LIST
    ]
    gt_filename_list = sorted(gt_filename_list)

    if len(gt_filename_list) > 0:
        logging.info(f"Found {len(gt_filename_list)} GT Maps")
    else:
        logging.error(f"No GT Maps found in '{gt_dir}")
        exit(1)

    # Associate files




    exit(0)
