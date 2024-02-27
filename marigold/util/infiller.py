import argparse
import os
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

plt.interactive(False)


class DepthPreprocessor():
    def __init__(self, apply_normalization: bool= False, depth_scale: float = 256.0, sky_removal: bool=False):
        self.depth_scale = depth_scale
        self.apply_normalization = apply_normalization
        ...

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Fill missing values using a Nearest Neighbor approach
        indices = distance_transform_edt(image == 0, return_distances=False, return_indices=True)
        image = image[indices[0], indices[1]]
        if self.apply_normalization:
            # Cast to meters [for kitti meters = (float)image / 256.0] and compute 2% and 98% percentiles
            image = image.astype(np.float32) / self.depth_scale
            thresh_min = np.percentile(image, 2)
            thresh_max = np.percentile(image, 98)
            # print(f'Inpainted image: [{thresh_min} - {thresh_max}]')
            image = 2 * ((image - thresh_min) / (thresh_max - thresh_min) - 0.5)
        return image

def read_depth(filename: str) -> np.uint16:
    return cv2.imread(filename, cv2.IMREAD_UNCHANGED)

EXTENSION_LIST = ['.tiff', '.png']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Applies Nearest-Neighbor infilling to an input set of depth images')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--depth-scale', type=float, required=True)

    args = parser.parse_args()

    preprocessor = DepthPreprocessor(apply_normalization=False, depth_scale=args.depth_scale, sky_removal=False)

    os.makedirs(args.output_dir, exist_ok=True)
    # Load input maps
    input_depths_path = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in EXTENSION_LIST]
    input_depths = [read_depth(os.path.join(args.input_dir, f)) for f in input_depths_path]
    output_depths = (preprocessor(x) for x in input_depths)
    for filename, image in tqdm(zip(input_depths_path, output_depths), total=len(input_depths_path)):
        cv2.imwrite(os.path.join(args.output_dir, filename), image)

    exit(0)
    # depth_preproc = DepthPreprocessor(depth_scale=1000)
    # input_image = read_depth(IM1_PATH)
    # image = depth_preproc(input_image)
    # fig, axs = plt.subplots(2, 1)
    # axs[0].imshow(input_image)
    # axs[0].set_title('Input')
    # axs[1].imshow(image)
    # axs[1].set_title('Post-Processed')
    # plt.show()
