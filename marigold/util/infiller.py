import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

plt.interactive(False)
class DepthPreprocessor():
    def __init__(self, sky_removal: bool = False, depth_scale: float = 256.0):
        self.depth_scale = depth_scale
        ...

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Fill missing values using a Nearest Neighbor approach
        indices = distance_transform_edt(image == 0, return_distances=False, return_indices=True)
        image = image[indices[0], indices[1]]
        # Cast to meters [for kitti meters = (float)image / 256.0] and compute 2% and 98% percentiles
        image = image.astype(np.float32) / self.depth_scale
        thresh_min = np.percentile(image, 2)
        thresh_max = np.percentile(image, 98)
        print(f'Inpainted image: [{thresh_min} - {thresh_max}]')
        image = 2 * ((image - thresh_min) / (thresh_max - thresh_min) - 0.5)
        return image


def infill_image(image: np.ndarray) -> np.ndarray:
    return cv2.inpaint(image, (image == 0).astype(np.uint8), 1, cv2.INPAINT_TELEA)

def read_depth(filename: str) -> np.uint16:
    return cv2.imread(filename, cv2.IMREAD_UNCHANGED)



IM_PATH = '/home/eg/data/kitti_depth_completion/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02/0000000005.png'
IM1_PATH = '/home/eg/data/vbr_campus/campus1_short_kitti/cloud_reprojection/1946.3325201.tiff'
if __name__ == '__main__':
    depth_preproc = DepthPreprocessor(depth_scale=1000)
    input_image = read_depth(IM1_PATH)
    image = depth_preproc(input_image)
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(input_image)
    axs[0].set_title('Input')
    axs[1].imshow(image)
    axs[1].set_title('Post-Processed')
    plt.show()


