import argparse
import logging
import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

EXTENSION_LIST = [".tiff", ".png", ".npy"]

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
        default="viridis",
        help="Colormap used to render error maps"
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    gt_dir = args.gt_dir
    gt_scale = args.gt_scale
    output_dir = args.output_dir

    # Output Directories
    output_dir_error = os.path.join(output_dir, "error_l1")
    output_dir_error_npy = os.path.join(output_dir, "error_l1_npy")
    output_dir_abs = os.path.join(output_dir, "aligned")
    output_dir_abs_npy = os.path.join(output_dir, "aligned_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_error, exist_ok=True)
    os.makedirs(output_dir_abs, exist_ok=True)
    os.makedirs(output_dir_error_npy, exist_ok=True)
    os.makedirs(output_dir_abs_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # Data
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

    gt_filename_list = glob(os.path.join(gt_dir, "*"))
    gt_filename_list = [
        f for f in gt_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    gt_filename_list = sorted(gt_filename_list)

    if len(gt_filename_list) > 0:
        logging.info(f"Found {len(gt_filename_list)} GT Maps")
    else:
        logging.error(f"No GT Maps found in '{gt_dir}")
        exit(1)

    # Associate files
    matching_pairs = [
        (est_f, gt_f)
        for est_f in input_filename_list
        for gt_f in gt_filename_list
        if os.path.splitext(os.path.basename(est_f))[0][:-5] == os.path.splitext(os.path.basename(gt_f))[0]
    ]

    num_pairs = len(matching_pairs)
    if num_pairs > 0:
        logging.info(f"Found {num_pairs} matching pairs")
    else:
        logging.error(f"Found no matching pairs between input sets")
        exit(1)

    mae_lst = []
    for input_filename, gt_filename in tqdm(matching_pairs, desc="Evaluating depth-maps"):
        if input_filename.endswith(".npy"):
            input_image = np.load(input_filename)
        else:
            input_image = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)

        if gt_filename.endswith(".npy"):
            gt_image = np.load(gt_filename)
        else:
            gt_image = cv2.imread(gt_filename, cv2.IMREAD_UNCHANGED)

        # Create mask for evaluation
        mask = gt_image > 0
        input_image[~mask] = 0

        # Bring GT to meters
        gt_image = gt_image.astype(np.float32) / gt_scale

        # Relative-Depth -> Absolute-Depth using GT to align
        # Solve (a, b) * A = b
        # A = matrix(2, N) | A[0, :] = relative_depths [1 x N] | A[1, :] = 1 [1 x N]
        # b = GT_depths [1 x N]

        valid_gt = gt_image[mask]
        valid_depths = input_image[mask]

        A = np.vstack((valid_depths,
                       np.ones_like(valid_depths))).T
        a, b = np.linalg.lstsq(A, valid_gt, rcond=-1)[0]

        input_image[mask] = input_image[mask] * a + b

        # Compute error map
        error_map = np.abs(input_image - gt_image)

        # fig, axs = plt.subplots(3,1, figsize=(10, 30))
        # axs[0].imshow(input_image, cmap=args.color_map.lower())
        # axs[1].imshow(gt_image, cmap=args.color_map.lower())
        # im_err = axs[2].imshow(error_map, cmap=args.color_map.lower())
        # divider = make_axes_locatable(axs[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # fig.colorbar(im_err, cax=cax, orientation="vertical")
        # axs[0].set_title("Input")
        # axs[1].set_title("Ground-Truth")
        # axs[2].set_title("Post-alignment L1 norm")

        error = error_map[mask]
        mae = np.mean(error)
        mae_lst.append((os.path.splitext(os.path.basename(input_filename))[0], mae))

        # Save input_depth [in aligned] and error_map [in error_l1]
        # Save also numpy versions for further processing

        np.save(os.path.join(output_dir_abs_npy, os.path.basename(input_filename)), input_image)
        np.save(os.path.join(output_dir_error_npy, os.path.basename(input_filename)), error_map)
        # Normalize images to color them and save
        input_image = (input_image / np.max(input_image) * 255).astype(np.uint8)
        error_map = (error_map / np.max(error_map) * 255).astype(np.uint8)
        input_image = cv2.applyColorMap(input_image, cv2.COLORMAP_VIRIDIS)
        error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_VIRIDIS)

        cv2.imwrite(os.path.join(output_dir_abs, os.path.splitext(os.path.basename(input_filename))[0] + ".png"),
                    input_image)
        cv2.imwrite(os.path.join(output_dir_error, os.path.splitext(os.path.basename(input_filename))[0] + ".png"),
                    error_map)
        # print(f"MAE: {np.sum(error) / len(error)} | MAX_E: {np.max(error)}")
        # plt.tight_layout()
        # plt.show()
    mae_set = np.asarray([b for a, b in mae_lst])
    mae_filename = "results.csv"
    with open(os.path.join(output_dir, mae_filename), "w") as f:
        f.write("filename,mae\n")
        for fname, mae in mae_lst:
            f.write(f"{fname},{mae}\n")
    print(f"Average MAE: {np.mean(mae_set)}")

    exit(0)
